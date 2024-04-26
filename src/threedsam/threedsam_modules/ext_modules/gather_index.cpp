#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> gather_index_cuda(
    torch::Tensor source,
    const int64_t collect_size
);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> gather_index(
    const torch::Tensor source,
    const int64_t collect_size) {
        CHECK_INPUT(source);

        return gather_index_cuda(source, collect_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gather_index", &gather_index, "Gather variable-length index where value is true.");
}