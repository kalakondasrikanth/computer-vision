cmake_minimum_required(VERSION 2.8 )

project(lab6)

set(CMAKE_BUILD_TYPE RelWithDebInfo)

add_definitions("-std=c++11")

find_package(OpenCV REQUIRED)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  )

add_library(${PROJECT_NAME}_ObjDetId
  include/ObjectDetection.h
  src/ObjectDetection.cpp
  )

target_link_libraries(${PROJECT_NAME}_ObjDetId
  ${OpenCV_LIBS}
  )

add_executable(${PROJECT_NAME}_test
  src/main.cpp
  )
target_link_libraries(${PROJECT_NAME}_test
  ${OpenCV_LIBS}
  ${PROJECT_NAME}_classifier
  )

