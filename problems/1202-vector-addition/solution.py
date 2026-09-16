#include <cuda_runtime.h>
#include <vector>

__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    // c[i] = a[i] + b[i], guarded by i < n
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        c[i] = a[i] + b[i];
    }
}

std::vector<float> vector_add(const std::vector<float>& a, const std::vector<float>& b) {
    // allocate a, b, c on the device; copy a and b over (Host -> Device);
    // launch the kernel; copy c back; free memory
    
    float *d_a, *d_b, *d_c;

    int n = static_cast<int>(a.size());
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));

    cudaMemcpy(d_a, a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerblock = 256;

    int blocks = (n + threadsPerblock -1) / threadsPerblock;

    add_kernel<<<blocks,threadsPerblock>>>(d_a, d_b, d_c, n);
    std::vector<float> result(n);
    
    cudaMemcpy(result.data(), d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    // cudafree 
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return result;
}
