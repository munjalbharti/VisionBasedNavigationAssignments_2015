cmake_minimum_required(VERSION 2.8.3)
project(euroc_comm)

find_package(catkin REQUIRED COMPONENTS 
  message_generation 
  geometry_msgs 
  sensor_msgs 
  octomap_msgs)

add_message_files(
  FILES
  VisualInertial.msg
  VisualInertialWithPose.msg
)

add_service_files(
  FILES
  Task1.srv
  Task2.srv
)

generate_messages(
  DEPENDENCIES 
  geometry_msgs 
  sensor_msgs 
  octomap_msgs
)

catkin_package(
  DEPENDS
  INCLUDE_DIRS 
  CATKIN_DEPENDS 
    message_runtime 
    geometry_msgs 
    sensor_msgs 
    octomap_msgs
  LIBRARIES
)

