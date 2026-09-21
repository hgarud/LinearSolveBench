cmake_minimum_required(VERSION 3.20)
project(linear_solver_bench_driver LANGUAGES C CXX)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_path(HYPRE_INCLUDE_DIR HYPRE.h REQUIRED)
set(LSB_RUNTIME_ROOT "" CACHE PATH "Runtime root to normalize in compiled diagnostics")

add_library(lsb_driver OBJECT src/driver.cpp)
target_include_directories(lsb_driver PRIVATE ${HYPRE_INCLUDE_DIR})
target_compile_options(lsb_driver PRIVATE -O3 -DNDEBUG -fno-omit-frame-pointer)
target_compile_options(lsb_driver PRIVATE
    "-ffile-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}=/linear-solver-bench/native")
if(LSB_RUNTIME_ROOT)
    target_compile_options(lsb_driver PRIVATE
        "-ffile-prefix-map=${LSB_RUNTIME_ROOT}=/linear-solver-bench/runtime")
endif()

install(FILES $<TARGET_OBJECTS:lsb_driver> DESTINATION trusted RENAME driver.o)
