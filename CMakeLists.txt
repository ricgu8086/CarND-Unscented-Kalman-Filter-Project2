project(UnscentedKF)

cmake_minimum_required (VERSION 3.5)

add_definitions(-std=c++0x)

set(CXX_FLAGS "-Wall -g3")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

set(sources
   src/ukf.cpp
   src/main.cpp
   src/tools.cpp)

add_executable(UnscentedKF ${sources})
