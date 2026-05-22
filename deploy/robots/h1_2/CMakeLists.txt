cmake_minimum_required(VERSION 3.12)
project(h1_2_controller)

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(yaml-cpp REQUIRED)

include_directories(
  /usr/include/eigen3
  /opt/unitree_robotics/include # unitree_sdk2 installed here
  /opt/unitree_robotics/include/ddscxx # dds cyclonedds-cxx
  /usr/local/include/ddscxx # fallback
  /usr/local/include/iceoryx/v2.0.2 # dds
  include
  ${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0/include
  ${PROJECT_SOURCE_DIR}/../../include/
)

link_directories(/opt/unitree_robotics/lib)

link_libraries(
  unitree_sdk2 ddsc ddscxx rt pthread # dds
  libboost_program_options.a libyaml-cpp.a fmt
  ${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so
)

file(GLOB_RECURSE ADD_SRC_LIST
  ${PROJECT_SOURCE_DIR}/src/*.cpp
)

add_library(${PROJECT_NAME}_lib ${ADD_SRC_LIST})
link_libraries(${PROJECT_NAME}_lib)

add_executable(h1_2_ctrl main.cpp)
