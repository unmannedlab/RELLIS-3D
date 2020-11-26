cmake_minimum_required(VERSION 2.8.3)
project(platform_description)

find_package(catkin REQUIRED COMPONENTS roslaunch)

catkin_package()

roslaunch_add_file_check(launch/description.launch)

install(DIRECTORY meshes launch urdf
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(PROGRAMS scripts/env_run
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
