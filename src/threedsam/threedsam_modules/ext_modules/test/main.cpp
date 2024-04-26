#include <stdio.h>
#include <iostream>
#include <torch/torch.h>
#include <cuda_runtime.h>
#include <vector>

// #define checkRuntime(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__);

// bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line) {
//     if (code != cudaSuccess) {
//         const char* errorName = cudaGetErrorName(code);
//         const char* errorMessage = cudaGetErrorString(code);
//         printf("runtime error %s: %d, %s failed. \n code = %s, message = %s", file, line, op, errorName, errorMessage);
//         return false;
//     }    
//     return true;
// }

std::vector<torch::Tensor> gather_index(
    const torch::Tensor source,
    const int64_t collect_size
);

int main(int argc, char* argv[]) {
    int size[3] = {0};
    for (int i = 1; i < 4; i++) {
        size[i - 1] = atoi(argv[i]);
    }
    int max_num = atoi(argv[4]);

    torch::Device device(torch::kCUDA);
    torch::Tensor mask = torch::rand({size[0], size[1], size[2]}, device);
    float threshold = 0.5;
    auto bool_mask = mask > 0.5;
    // bool_mask[0][0][0] = true;
    std::cout << "source mask: " << bool_mask << std::endl;

    std::vector<torch::Tensor> result = gather_index(bool_mask, max_num); 
    std::cout << "Done." << std::endl;  
}