cmake_minimum_required(VERSION 3.20)
project(numpyLearch)
set(CMAKE_CXX_STANDARD 14)

include_directories(${NUMCPP_INCLUDE_DIR})
include_directories(${BOOST_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include)
add_executable(numpyLearch main.cpp)