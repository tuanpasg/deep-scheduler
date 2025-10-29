cmake_minimum_required(VERSION 3.10)
project(ppo_onnx_bench C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Set ONNXRUNTIME_DIR to the root where include/ and lib/ reside
# e.g., -DONNXRUNTIME_DIR=/path/to/onnxruntime-linux-x64-<ver>
if(NOT DEFINED ONNXRUNTIME_DIR)
  message(FATAL_ERROR "Please set -DONNXRUNTIME_DIR to your ONNX Runtime unpacked directory")
endif()

include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

add_executable(ppo_onnx_bench main.c)
# Static link if available; otherwise onnxruntime
target_link_libraries(ppo_onnx_bench onnxruntime)

# On Linux, you may want: -pthread and -Wl,-rpath to find the shared lib at runtime
if(UNIX AND NOT APPLE)
  target_link_libraries(ppo_onnx_bench pthread)
  set_target_properties(ppo_onnx_bench PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_DIR}/lib"
    INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib")
endif()
