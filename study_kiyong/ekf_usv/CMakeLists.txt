cmake_minimum_required(VERSION 3.0.2)
project(ekf_usv)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  Eigen3 REQUIRED
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

add_executable(ekf_usv src/ekf_filter.cpp)
# add_executable(ekf_usv src/ekf_backup.cpp)

target_link_libraries(ekf_usv
  ${catkin_LIBRARIES}
)