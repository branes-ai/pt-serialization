cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(direct_pytorch_parser)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set path for libtorch
set(CMAKE_PREFIX_PATH "C:/iree/libtorch")

# Find PyTorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Verify PyTorch installation
if(NOT TORCH_FOUND)
    message(FATAL_ERROR "PyTorch not found. Please check LibTorch installation path.")
endif()

message(STATUS "LibTorch version: ${Torch_VERSION}")
message(STATUS "LibTorch libraries: ${TORCH_LIBRARIES}")
message(STATUS "LibTorch install prefix: ${TORCH_INSTALL_PREFIX}")

# Create executable
add_executable(pytorch_parser 
    main.cpp
)

# Link PyTorch libraries
target_link_libraries(pytorch_parser 
    ${TORCH_LIBRARIES}
)

# The following flag is needed on Windows
if(WIN32)
    set_property(TARGET pytorch_parser PROPERTY
                 CXX_STANDARD 17)
    
    # Copy PyTorch shared libraries to output directory (Windows)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    if(TORCH_DLLS)
        add_custom_command(TARGET pytorch_parser
                           POST_BUILD
                           COMMAND ${CMAKE_COMMAND} -E copy_if_different
                           ${TORCH_DLLS}
                           $<TARGET_FILE_DIR:pytorch_parser>)
        message(STATUS "Will copy PyTorch DLLs: ${TORCH_DLLS}")
    endif()
endif()

# Set target properties for MSVC runtime library
if(MSVC)
    set_property(TARGET pytorch_parser PROPERTY
                 MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

# Add some helpful build information
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")

# Optional: Add compiler-specific optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        target_compile_options(pytorch_parser PRIVATE /O2)
    else()
        target_compile_options(pytorch_parser PRIVATE -O3 -march=native)
    endif()
endif()

# Optional: Enable parallel builds for MSVC
if(MSVC)
    target_compile_options(pytorch_parser PRIVATE /MP)
endif()

# Print final configuration
message(STATUS "")
message(STATUS "===================== BUILD CONFIGURATION =====================")
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Target: pytorch_parser")
message(STATUS "LibTorch version: ${Torch_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "==============================================================")