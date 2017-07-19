cmake_minimum_required(VERSION 2.8)
project(BlurImage)
find_package(OpenCV REQUIRED)
find_package(CUDA)

set(INC_LOCAL_DIR /usr/local/include/ /usr/local/cuda/include/) 
include_directories(${INC_LOCAL_DIR}) 

SET(CMAKE_C_COMPILER g++)
SET(CMAKE_BUILD_TYPE "Debug") 
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
add_compile_options(-std=c++11)

file( GLOB  hdr *.hpp *.h )
file( GLOB  cu  *.cu)
SET (HW2_files main.cpp reference_calc.cpp compare.cpp)
    
CUDA_ADD_EXECUTABLE(HW2 ${HW2_files} ${hdr} ${cu})

target_link_libraries(HW2 ${OpenCV_LIBS})
