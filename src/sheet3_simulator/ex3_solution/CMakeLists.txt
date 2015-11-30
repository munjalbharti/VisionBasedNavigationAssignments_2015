cmake_minimum_required(VERSION 2.8.3)
project(ex3_solution)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  mav_msgs
  roscpp
  sensor_msgs
  tf_conversions
  eigen_conversions
)

find_package(cmake_modules REQUIRED)
find_package(Eigen REQUIRED)
find_package(sophus REQUIRED)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
)

catkin_package(
  DEPENDS sophus eigen_conversions tf_conversions
  CATKIN_DEPENDS geometry_msgs mav_msgs roscpp sensor_msgs
  )

# Extend with executables for the subtasks, if necessary.
add_executable(${PROJECT_NAME} src/solution.cpp)
add_dependencies(${PROJECT_NAME} ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES})
