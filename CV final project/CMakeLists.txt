cmake_minimum_required(VERSION 2.8.1)

project(CV_FINAL)
add_definitions("-std=c++11")

find_package(OpenCV REQUIRED)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS} )

include_directories(
  src
  ${OpenCV_INCLUDE_DIRS} )

add_library(${PROJECT_NAME}_lib
  include/TemplateMatching.h
  include/svmbased.h
  src/TemplateMatching.cpp
  src/svmbased.cpp
  src/svmbased2.cpp)

target_link_libraries(${PROJECT_NAME}_lib ${OpenCV_LIBS} )

add_executable(${PROJECT_NAME}_FP
  src/main.cpp)

target_link_libraries(${PROJECT_NAME}_FP ${OpenCV_LIBS}  ${PROJECT_NAME}_lib)

