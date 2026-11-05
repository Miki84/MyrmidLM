# PrivateLM Linux node: rpc-server, llama-server and privatelm-probe, built
# from the llama.cpp commit the app vendors (see LLAMA_CPP_COMMIT), so the
# RPC protocol matches the phones exactly.
#
# Normally driven by build.sh. Backends (CUDA, HIP/ROCm, Vulkan, SYCL) are
# built as runtime-loaded modules next to the binaries (GGML_BACKEND_DL), so
# one install can carry several GPU backends and still start on a machine
# that lacks one of them.
cmake_minimum_required(VERSION 3.22)
project(privatelm_node C CXX)

set(LLAMA_CPP_DIR "" CACHE PATH "llama.cpp source tree at the pinned commit")
if(NOT EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "Set LLAMA_CPP_DIR to the llama.cpp checkout (build.sh does this)")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
set(CMAKE_BUILD_RPATH "\$ORIGIN")
set(CMAKE_INSTALL_RPATH "\$ORIGIN")

set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
set(GGML_RPC ON CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL ON CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_CPU_ALL_VARIANTS ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)

add_subdirectory(${LLAMA_CPP_DIR} llama.cpp)

add_executable(privatelm-probe probe/privatelm-probe.cpp)
target_link_libraries(privatelm-probe PRIVATE ggml ggml-base)
target_compile_features(privatelm-probe PRIVATE cxx_std_17)
