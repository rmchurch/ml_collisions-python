cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

#debugging (doesnt work?)
#variable_watch(CUDNN_LIBRARY)

#set(TENSORRT_ROOT  "/scratch/gpfs/rmc2/conda-envs/torch-env/" CACHE STRING "tensorrt_root")
add_executable(test_model test_model.cpp)
target_link_libraries(test_model "${TORCH_LIBRARIES}")
set_property(TARGET test_model PROPERTY CXX_STANDARD 14)
