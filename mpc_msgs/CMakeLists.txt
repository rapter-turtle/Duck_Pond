cmake_minimum_required(VERSION 2.8.3)
project(mpc_msgs)
find_package(catkin REQUIRED
  message_generation
  std_msgs
  geometry_msgs
)

add_message_files( DIRECTORY msg
  FILES
  MPCTraj.msg
  MPC_State.msg  
  Obs_State.msg  
)

generate_messages(DEPENDENCIES std_msgs)
catkin_package(CATKIN_DEPENDS message_runtime std_msgs geometry_msgs)
