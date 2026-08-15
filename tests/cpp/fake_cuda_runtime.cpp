#include <cstdlib>
#include <cstring>

extern "C" {

int cudaGetDeviceCount(int* count) {
    if (count == nullptr) return 1;
    *count = 1;
    return 0;
}
int cudaSetDevice(int device) { return device == 0 ? 0 : 1; }
int cudaMemGetInfo(std::size_t* free_bytes, std::size_t* total_bytes) {
    if (free_bytes == nullptr || total_bytes == nullptr) return 1;
    *total_bytes = static_cast<std::size_t>(16ULL * 1024ULL * 1024ULL * 1024ULL);
    *free_bytes = static_cast<std::size_t>(15ULL * 1024ULL * 1024ULL * 1024ULL);
    return 0;
}
int cudaMalloc(void** pointer, std::size_t bytes) {
    if (pointer == nullptr) return 1;
    *pointer = std::malloc(bytes == 0 ? 1 : bytes);
    return *pointer == nullptr ? 2 : 0;
}
int cudaFree(void* pointer) { std::free(pointer); return 0; }
int cudaMemcpy(void* target, const void* source, std::size_t bytes, int) {
    if (target == nullptr || source == nullptr) return 1;
    std::memmove(target, source, bytes);
    return 0;
}
int cudaMemcpyAsync(
    void* target, const void* source, std::size_t bytes, int kind, void*) {
    return cudaMemcpy(target, source, bytes, kind);
}
int cudaMemsetAsync(void* target, int value, std::size_t bytes, void*) {
    if (target == nullptr) return 1;
    std::memset(target, value, bytes);
    return 0;
}
int cudaStreamCreate(void** stream) {
    if (stream == nullptr) return 1;
    *stream = std::malloc(1);
    return *stream == nullptr ? 2 : 0;
}
int cudaStreamDestroy(void* stream) { std::free(stream); return 0; }
int cudaStreamSynchronize(void*) { return 0; }
int cudaStreamWaitEvent(void*, void*, unsigned int) { return 0; }
int cudaEventCreateWithFlags(void** event, unsigned int) {
    if (event == nullptr) return 1;
    *event = std::malloc(1);
    return *event == nullptr ? 2 : 0;
}
int cudaEventDestroy(void* event) { std::free(event); return 0; }
int cudaEventRecord(void*, void*) { return 0; }
const char* cudaGetErrorString(int status) {
    return status == 0 ? "success" : "fake CUDA failure";
}

}
