cmake_minimum_required(VERSION 3.5)

project(car_assembly_2nd)

find_package(ament_cmake REQUIRED)

ament_environment_hooks(
  env-hooks/gazebo_model_path.dsv
)

install(
  DIRECTORY config launch meshes textures urdf worlds
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS scripts/gesture_gazebo_driver
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
