#include <cuda_runtime.h>
#include <vector>

__global__ void dot_kernel(const float* a, const float* b, float* out, int n) {
    // load a[tid]*b[tid] (or 0) into shared memory, then tree-reduce and write out[0]
    __shared__ float shared[256];

    int tid = threadIdx.x;
    
    // 每个线程一个乘积
    if (tid < n){
        shared[tid] = a[tid] * b[tid];
    }else{
        shared[tid] = 0.0f;
    }

    __syncthreads();

    // 共享内存规约
    for (int stride = blockDim.x / 2 ; stride > 0; stride /= 2){
        if (tid < stride){
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    // 线程0负责最终的和
    if (tid == 0){
        *out = shared[0];
    }
}

float dot_product(const std::vector<float>& a, const std::vector<float>& b) {
    int n = static_cast<int>(a.size());
    float *d_a, *d_b, *d_result;

    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));

    cudaMemcpy(d_a, a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    // n <= 256, so use one block 
    int threads = 256;
    
    dot_kernel<<<1, threads>>>(d_a, d_b, d_result, n);

    float result;
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // CUDA内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    return result;
}
