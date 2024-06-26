# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# compile CUDA with /usr/local/cuda-12.2/bin/nvcc
# compile CXX with /usr/bin/c++
CUDA_DEFINES = -DUSE_C10D_GLOO -DUSE_C10D_NCCL -DUSE_DISTRIBUTED -DUSE_RPC -DUSE_TENSORPIPE

CUDA_INCLUDES = --options-file CMakeFiles/exec.dir/includes_CUDA.rsp

CUDA_FLAGS =  -DONNX_NAMESPACE=onnx_c2 -gencode arch=compute_86,code=sm_86 -Xcudafe --diag_suppress=cc_clobber_ignored,--diag_suppress=set_but_not_used,--diag_suppress=field_without_dll_interface,--diag_suppress=base_class_has_different_dll_interface,--diag_suppress=dll_interface_conflict_none_assumed,--diag_suppress=dll_interface_conflict_dllexport_assumed,--diag_suppress=bad_friend_decl --expt-relaxed-constexpr --expt-extended-lambda -g -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0

CXX_DEFINES = -DUSE_C10D_GLOO -DUSE_C10D_NCCL -DUSE_DISTRIBUTED -DUSE_RPC -DUSE_TENSORPIPE

CXX_INCLUDES = -isystem /usr/local/cuda-12.2/include -isystem /home/morgen/.conda/envs/3dsam/lib/python3.8/site-packages/torch/include -isystem /home/morgen/.conda/envs/3dsam/lib/python3.8/site-packages/torch/include/torch/csrc/api/include

CXX_FLAGS = -g -std=gnu++17 -D_GLIBCXX_USE_CXX11_ABI=0

