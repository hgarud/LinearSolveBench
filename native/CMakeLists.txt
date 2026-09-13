cmake_minimum_required(VERSION 3.20)
project(linear_solver_bench_driver LANGUAGES C CXX)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_path(HYPRE_INCLUDE_DIR HYPRE.h REQUIRED)
find_library(HYPRE_LIBRARY NAMES HYPRE REQUIRED)

add_library(lsb_driver OBJECT src/driver.cpp)
target_include_directories(lsb_driver PRIVATE ${HYPRE_INCLUDE_DIR})
target_compile_options(lsb_driver PRIVATE -O3 -DNDEBUG -fno-omit-frame-pointer)
