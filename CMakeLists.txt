project(pc_sense)
cmake_minimum_required(VERSION 2.8.3)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
find_package(catkin_simple REQUIRED)

catkin_simple()
find_package(OpenCV REQUIRED)
find_package(PCL  REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} )
add_definitions(${PCL_DEFINITIONS})

set(HEADERS
  ./include/pc_sense/pc_op.hpp
)

set(SOURCES
  ./src/pc_op.cpp
)


cs_add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES} Eigen3::Eigen)

cs_add_executable(pc_project_ros src/pc_project_ros.cpp)
target_link_libraries(pc_project_ros ${PROJECT_NAME})

cs_install()
cs_export()