#include <torch/torch.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <stdio.h>

#define checkRuntime(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__)
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

bool __check_cuda_runtime(cudaError_t code, const char *op, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        const char *err_name = cudaGetErrorName(code);
        const char *err_message = cudaGetErrorString(code);
        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);
        return false;
    }
    return true;
}

__global__ void gather_index_cuda_kernel(
    const torch::PackedTensorAccessor32<bool, 3, torch::RestrictPtrTraits> source,
    torch::PackedTensorAccessor32<int64_t, 3, torch::RestrictPtrTraits> output_index,
    torch::PackedTensorAccessor32<bool, 3, torch::RestrictPtrTraits> output_mask,
    const int64_t ROW_SIZE, const int64_t COL_SIZE, const int64_t COLLECT_SIZE)
{
    // batch index
    const int n = blockIdx.y;
    // row index
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    int count = 0;
    if (row < ROW_SIZE)
    {
        for (int col = 0; col < COL_SIZE; col++)
        {
            if (count >= COLLECT_SIZE)
                break;
            if (source[n][row][col] == true)
            {
                output_index[n][row][count] = col;
                output_mask[n][row][count] = true;
                ++count;
            }
        }
    }
}

std::vector<torch::Tensor> gather_index_cuda(
    const torch::Tensor source,
    const int64_t collect_size)
{
    const auto source_size = source.sizes();
    const auto batch_size = source_size[0];
    const auto row_size = source_size[1];
    const auto col_size = source_size[2];

    torch::Device device(torch::kCUDA);
    auto output_mask = torch::zeros(
        {source_size[0], source_size[1], collect_size},
        torch::dtype(torch::kBool).device(torch::kCUDA, 0));
    auto output_index = torch::zeros(
        {source_size[0], source_size[1], collect_size},
        torch::dtype(torch::kInt64).device(torch::kCUDA, 0));

    const int threads = 1024;
    const dim3 blocks((row_size + threads - 1) / threads, batch_size);
    gather_index_cuda_kernel<<<blocks, threads>>>(
        source.packed_accessor32<bool, 3, torch::RestrictPtrTraits>(),
        output_index.packed_accessor32<int64_t, 3, torch::RestrictPtrTraits>(),
        output_mask.packed_accessor32<bool, 3, torch::RestrictPtrTraits>(),
        row_size, col_size, collect_size);
    checkRuntime(cudaDeviceSynchronize());
    // checkRuntime(cudaGetLastError());
    return {output_index, output_mask};
}

std::vector<torch::Tensor> gather_index(
    const torch::Tensor source,
    const int64_t collect_size)
{
    CHECK_INPUT(source);

    return gather_index_cuda(source, collect_size);
}