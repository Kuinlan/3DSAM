cmake_minimum_required(VERSION 3.20)
project(main)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_STANDARD 17)
set(Torch_DIR  "/home/morgen/.conda/envs/3dsam/lib/python3.8/site-packages/torch/share/cmake/Torch")

find_package(CUDA REQUIRED)
find_package(Torch REQUIRED)

# include_directories(${TORCH_INCLUDE_DIRS})
# include_directories(${CUDA_INCLUDE_DIRS})

file(GLOB SOURCES ./*.cpp ./*.cu)

add_executable(exec ${SOURCES})

target_include_directories(exec PRIVATE ${CUDA_INCLUDE_DIRS})
target_include_directories(exec PRIVATE ${TORCH_INCLUDE_DIRS})

target_link_directories(exec PRIVATE ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
target_link_libraries(exec ${TORCH_LIBRARIES})