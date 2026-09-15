#include <cuda_runtime.h>
#include <vector>

__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    // c[i] = a[i] + b[i], guarded by i < n
    
}

std::vector<float> vector_add(const std::vector<float>& a, const std::vector<float>& b) {
    // allocate a, b, c on the device; copy a and b over (Host -> Device);
    // launch the kernel; copy c back; free memory
    return {};
}
