#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <unistd.h>

namespace fs = std::filesystem;

namespace {

struct UniqueId {
    std::array<char, 128> bytes{};
};

struct Communicator {
    int world = 0;
    int rank = 0;
    fs::path root;
    std::unordered_map<int, std::uint64_t> send_sequences;
    std::unordered_map<int, std::uint64_t> receive_sequences;
    std::uint64_t all_reduce_sequence = 0;
};

constexpr int kSuccess = 0;
constexpr int kInvalid = 1;
constexpr int kIo = 2;
constexpr int kTimeout = 3;
constexpr int kFloat32 = 7;
constexpr int kSum = 0;

std::string identifier(const UniqueId& id) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (std::size_t index = 0; index < 16; ++index) {
        out << std::setw(2)
            << static_cast<unsigned>(static_cast<unsigned char>(id.bytes[index]));
    }
    return out.str();
}

bool wait_for(const fs::path& path) {
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::seconds(30);
    while (std::chrono::steady_clock::now() < deadline) {
        if (fs::is_regular_file(path)) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return false;
}

bool write_atomic(const fs::path& path, const void* data, std::size_t bytes) {
    const fs::path temporary = path.string() + ".tmp-" +
        std::to_string(static_cast<long long>(::getpid()));
    {
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
        if (!out) return false;
    }
    std::error_code error;
    fs::rename(temporary, path, error);
    return !error;
}

fs::path message_path(
    const Communicator& comm, int source, int target, std::uint64_t sequence) {
    return comm.root /
        ("send-" + std::to_string(source) + "-" + std::to_string(target) +
         "-" + std::to_string(sequence) + ".bin");
}

fs::path reduce_path(
    const Communicator& comm, std::uint64_t sequence, int rank) {
    return comm.root /
        ("reduce-" + std::to_string(sequence) + "-" + std::to_string(rank) +
         ".bin");
}

fs::path ack_path(
    const Communicator& comm, std::uint64_t sequence, int rank) {
    return comm.root /
        ("ack-" + std::to_string(sequence) + "-" + std::to_string(rank));
}

}  // namespace

extern "C" {

int ncclGetUniqueId(UniqueId* id) {
    if (id == nullptr) return kInvalid;
    const auto now = static_cast<std::uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::uint64_t state = now ^
        (static_cast<std::uint64_t>(::getpid()) << 32U) ^ 0x9e3779b97f4a7c15ULL;
    for (std::size_t index = 0; index < id->bytes.size(); ++index) {
        state ^= state >> 12U;
        state ^= state << 25U;
        state ^= state >> 27U;
        state *= 0x2545f4914f6cdd1dULL;
        id->bytes[index] = static_cast<char>(state >> 56U);
    }
    return kSuccess;
}

int ncclCommInitRank(void** result, int world, UniqueId id, int rank) {
    if (result == nullptr || world <= 0 || rank < 0 || rank >= world) {
        return kInvalid;
    }
    auto* comm = new Communicator;
    comm->world = world;
    comm->rank = rank;
    comm->root = fs::temp_directory_path() /
        ("nfn-fake-nccl-" + identifier(id));
    std::error_code error;
    fs::create_directories(comm->root, error);
    if (error) {
        delete comm;
        return kIo;
    }
    *result = comm;
    return kSuccess;
}

int ncclCommDestroy(void* raw) {
    delete static_cast<Communicator*>(raw);
    return kSuccess;
}

int ncclSend(
    const void* data, std::size_t count, int datatype, int peer,
    void* raw, void*) {
    auto* comm = static_cast<Communicator*>(raw);
    if (comm == nullptr || data == nullptr || count == 0 || datatype != kFloat32 ||
        peer < 0 || peer >= comm->world || peer == comm->rank) {
        return kInvalid;
    }
    const auto sequence = comm->send_sequences[peer]++;
    return write_atomic(
        message_path(*comm, comm->rank, peer, sequence), data,
        count * sizeof(float)) ? kSuccess : kIo;
}

int ncclRecv(
    void* data, std::size_t count, int datatype, int peer,
    void* raw, void*) {
    auto* comm = static_cast<Communicator*>(raw);
    if (comm == nullptr || data == nullptr || count == 0 || datatype != kFloat32 ||
        peer < 0 || peer >= comm->world || peer == comm->rank) {
        return kInvalid;
    }
    const auto sequence = comm->receive_sequences[peer]++;
    const fs::path path = message_path(*comm, peer, comm->rank, sequence);
    if (!wait_for(path) || fs::file_size(path) != count * sizeof(float)) {
        return kTimeout;
    }
    std::ifstream in(path, std::ios::binary);
    in.read(static_cast<char*>(data), static_cast<std::streamsize>(count * sizeof(float)));
    if (!in || in.peek() != std::char_traits<char>::eof()) return kIo;
    std::error_code ignored;
    fs::remove(path, ignored);
    return kSuccess;
}

int ncclAllReduce(
    const void* send, void* receive, std::size_t count, int datatype, int operation,
    void* raw, void*) {
    auto* comm = static_cast<Communicator*>(raw);
    if (comm == nullptr || send == nullptr || receive == nullptr || count == 0 ||
        datatype != kFloat32 || operation != kSum) {
        return kInvalid;
    }
    const auto sequence = comm->all_reduce_sequence++;
    const fs::path own = reduce_path(*comm, sequence, comm->rank);
    if (!write_atomic(own, send, count * sizeof(float))) return kIo;
    for (int rank = 0; rank < comm->world; ++rank) {
        if (!wait_for(reduce_path(*comm, sequence, rank))) return kTimeout;
    }
    std::vector<float> total(count, 0.0f);
    std::vector<float> row(count);
    for (int rank = 0; rank < comm->world; ++rank) {
        const fs::path path = reduce_path(*comm, sequence, rank);
        if (fs::file_size(path) != count * sizeof(float)) return kIo;
        std::ifstream in(path, std::ios::binary);
        in.read(
            reinterpret_cast<char*>(row.data()),
            static_cast<std::streamsize>(row.size() * sizeof(float)));
        if (!in) return kIo;
        for (std::size_t index = 0; index < count; ++index) {
            total[index] += row[index];
        }
    }
    std::memcpy(receive, total.data(), count * sizeof(float));
    const char ack = 1;
    if (!write_atomic(ack_path(*comm, sequence, comm->rank), &ack, 1)) return kIo;
    if (comm->rank == 0) {
        for (int rank = 0; rank < comm->world; ++rank) {
            if (!wait_for(ack_path(*comm, sequence, rank))) return kTimeout;
        }
        std::error_code ignored;
        for (int rank = 0; rank < comm->world; ++rank) {
            fs::remove(reduce_path(*comm, sequence, rank), ignored);
            fs::remove(ack_path(*comm, sequence, rank), ignored);
        }
    }
    return kSuccess;
}

const char* ncclGetErrorString(int status) {
    switch (status) {
        case kSuccess: return "success";
        case kInvalid: return "invalid fake NCCL call";
        case kIo: return "fake NCCL I/O failure";
        case kTimeout: return "fake NCCL timeout";
        default: return "unknown fake NCCL failure";
    }
}

}  // extern "C"
