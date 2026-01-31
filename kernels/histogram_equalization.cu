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
#include <cub/cub.cuh>

#include "utils.cuh"

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

    unsigned char* d_in = img.data_ptr<unsigned char>();
    int* d_hist = hist_tensor.data_ptr<int>();
    int* d_cdf = cdf_tensor.data_ptr<int>();
    unsigned char* d_lut = lut_tensor.data_ptr<unsigned char>();
    unsigned char* d_out = result.data_ptr<unsigned char>();

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    cub::DeviceHistogram::HistogramEven(
        d_temp_storage, temp_storage_bytes,
        d_in, d_hist,
        256 + 1, 0, 256,
        pixels,
        stream
    );

    auto temp_storage_tensor = torch::empty(
        {static_cast<long>(temp_storage_bytes)}, 
        torch::TensorOptions().dtype(torch::kByte).device(img.device())
    );
    d_temp_storage = temp_storage_tensor.data_ptr();

    cub::DeviceHistogram::HistogramEven(
        d_temp_storage, temp_storage_bytes,
        d_in, d_hist,
        256 + 1, 0, 256,
        pixels,
        stream
    );

    auto policy = thrust::cuda::par.on(stream);

    thrust::inclusive_scan(policy, d_hist, d_hist + 256, d_cdf);

    int* d_cdf_min_ptr = thrust::min_element(policy, d_cdf, d_cdf + 256);
    
    int cdf_min;
    cudaMemcpyAsync(&cdf_min, d_cdf_min_ptr, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    float range = (float)(pixels - cdf_min);

    Normalize norm(cdf_min, range);
    
    auto transform_iter = thrust::make_transform_iterator(d_cdf, norm);
    
    thrust::copy(policy, transform_iter, transform_iter + 256, d_lut);

    int threads = 256;
    int blocks = (pixels + threads - 1) / threads;

    lut_kernel<<<blocks, threads, 0, stream>>>(
        d_in,
        d_out,
        d_lut,
        pixels
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}