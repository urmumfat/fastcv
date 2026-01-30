#include <c10/cuda/CUDAException.h> 
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/extrema.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/iterator/transform_iterator.h>
#include <algorithm>

#include "utils.cuh"


__global__ void histogram_kernel(const unsigned char* __restrict__ in, int* hist, int n) {
    __shared__ int temp_hist[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x < 256){
        temp_hist[threadIdx.x] = 0;
    }
    __syncthreads();

    if (idx < n) {
        atomicAdd(&temp_hist[in[idx]], 1);
    }
    __syncthreads();

    if (threadIdx.x < 256) {
        atomicAdd(&hist[threadIdx.x], temp_hist[threadIdx.x]);
    }
}

__global__ void lut_kernel(const unsigned char* __restrict__ in, unsigned char* __restrict__ out, const unsigned char* __restrict__ lut, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = lut[in[idx]];
    }
}


struct Normalize {
    int cdf_min;
    float range;
    
    __host__ __device__
    Normalize(int _min, float _range) : cdf_min(_min), range(_range) {}

    __host__ __device__
    unsigned char operator()(int cdf) const {
        float norm = ((float)cdf - cdf_min) / range * 255.0f;
        return (unsigned char)min(max(norm, 0.0f), 255.0f);
    }
};


torch::Tensor histogram_equalization(torch::Tensor img) {
    TORCH_CHECK(img.device().type() == torch::kCUDA, "Input image must be on CUDA");
    TORCH_CHECK(img.dtype() == torch::kByte, "Input image must be uint8");

    const int height = img.size(0);
    const int width = img.size(1);
    const int pixels = height * width;

    auto result = torch::empty_like(img);
    
    auto hist_tensor = torch::zeros({256}, torch::TensorOptions().dtype(torch::kInt32).device(img.device()));
    auto cdf_tensor = torch::empty({256}, torch::TensorOptions().dtype(torch::kInt32).device(img.device()));
    auto lut_tensor = torch::empty({256}, torch::TensorOptions().dtype(torch::kByte).device(img.device()));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int threads = 256;
    int blocks = (pixels + threads - 1) / threads;

    histogram_kernel<<<blocks, threads, 0, stream>>>(
        img.data_ptr<unsigned char>(),
        hist_tensor.data_ptr<int>(),
        pixels
    );

    thrust::device_ptr<int> d_hist_ptr(hist_tensor.data_ptr<int>());
    thrust::device_ptr<int> d_cdf_ptr(cdf_tensor.data_ptr<int>());
    thrust::device_ptr<unsigned char> d_lut_ptr(lut_tensor.data_ptr<unsigned char>());

    auto policy = thrust::cuda::par.on(stream);

    thrust::inclusive_scan(policy, d_hist_ptr, d_hist_ptr + 256, d_cdf_ptr);

    auto min_iter = thrust::min_element(policy, d_cdf_ptr, d_cdf_ptr + 256);
    
    int cdf_min;
    cudaMemcpyAsync(&cdf_min, thrust::raw_pointer_cast(&*min_iter), sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    float range = (float)(pixels - cdf_min);

    Normalize norm(cdf_min, range);
    auto transform_iter = thrust::make_transform_iterator(d_cdf_ptr, norm);
    
    thrust::copy(policy, transform_iter, transform_iter + 256, d_lut_ptr);

    lut_kernel<<<blocks, threads, 0, stream>>>(
        img.data_ptr<unsigned char>(),
        result.data_ptr<unsigned char>(),
        lut_tensor.data_ptr<unsigned char>(),
        pixels
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}