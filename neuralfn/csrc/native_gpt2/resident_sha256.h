#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>

namespace neuralfn::resident_support {

class Sha256 final {
public:
    Sha256()
        : state_{
              0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
              0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u} {}

    void update(const std::uint8_t* data, std::size_t size) {
        total_bytes_ += static_cast<std::uint64_t>(size);
        while (size > 0) {
            const std::size_t count = std::min(size, block_.size() - block_size_);
            std::memcpy(block_.data() + block_size_, data, count);
            block_size_ += count;
            data += count;
            size -= count;
            if (block_size_ == block_.size()) {
                transform(block_.data());
                block_size_ = 0;
            }
        }
    }

    std::string finish_hex() {
        const std::uint64_t bit_length = total_bytes_ * 8u;
        block_[block_size_++] = 0x80u;
        if (block_size_ > 56) {
            std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_size_), block_.end(), 0u);
            transform(block_.data());
            block_size_ = 0;
        }
        std::fill(
            block_.begin() + static_cast<std::ptrdiff_t>(block_size_),
            block_.begin() + 56,
            0u);
        for (std::size_t index = 0; index < 8; ++index) {
            block_[63 - index] = static_cast<std::uint8_t>(bit_length >> (index * 8));
        }
        transform(block_.data());
        std::ostringstream output;
        output << std::hex << std::setfill('0');
        for (std::uint32_t word : state_) {
            output << std::setw(8) << word;
        }
        return output.str();
    }

private:
    static std::uint32_t rotate_right(std::uint32_t value, std::uint32_t amount) {
        return (value >> amount) | (value << (32u - amount));
    }

    void transform(const std::uint8_t* block) {
        static constexpr std::array<std::uint32_t, 64> constants = {
            0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
            0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
            0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
            0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
            0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
            0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
            0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
            0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
            0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
            0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
            0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
            0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
            0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
            0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
            0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
            0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
        };
        std::array<std::uint32_t, 64> schedule{};
        for (std::size_t index = 0; index < 16; ++index) {
            const std::size_t offset = index * 4;
            schedule[index] =
                (static_cast<std::uint32_t>(block[offset]) << 24u) |
                (static_cast<std::uint32_t>(block[offset + 1]) << 16u) |
                (static_cast<std::uint32_t>(block[offset + 2]) << 8u) |
                static_cast<std::uint32_t>(block[offset + 3]);
        }
        for (std::size_t index = 16; index < schedule.size(); ++index) {
            const std::uint32_t small0 = rotate_right(schedule[index - 15], 7u) ^
                rotate_right(schedule[index - 15], 18u) ^ (schedule[index - 15] >> 3u);
            const std::uint32_t small1 = rotate_right(schedule[index - 2], 17u) ^
                rotate_right(schedule[index - 2], 19u) ^ (schedule[index - 2] >> 10u);
            schedule[index] = schedule[index - 16] + small0 + schedule[index - 7] + small1;
        }
        std::uint32_t a = state_[0];
        std::uint32_t b = state_[1];
        std::uint32_t c = state_[2];
        std::uint32_t d = state_[3];
        std::uint32_t e = state_[4];
        std::uint32_t f = state_[5];
        std::uint32_t g = state_[6];
        std::uint32_t h = state_[7];
        for (std::size_t index = 0; index < schedule.size(); ++index) {
            const std::uint32_t sum1 = rotate_right(e, 6u) ^
                rotate_right(e, 11u) ^ rotate_right(e, 25u);
            const std::uint32_t choose = (e & f) ^ ((~e) & g);
            const std::uint32_t first =
                h + sum1 + choose + constants[index] + schedule[index];
            const std::uint32_t sum0 = rotate_right(a, 2u) ^
                rotate_right(a, 13u) ^ rotate_right(a, 22u);
            const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t second = sum0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + first;
            d = c;
            c = b;
            b = a;
            a = first + second;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_{};
    std::array<std::uint8_t, 64> block_{};
    std::size_t block_size_ = 0;
    std::uint64_t total_bytes_ = 0;
};

inline std::string sha256_hex(const std::uint8_t* data, std::size_t size) {
    Sha256 digest;
    digest.update(data, size);
    return digest.finish_hex();
}

}  // namespace neuralfn::resident_support
