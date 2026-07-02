cmake_minimum_required(VERSION 3.5)
project(frenet_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/FrenetEgoState.msg"
  "msg/FrenetLocalTrajectory.msg"
  "msg/FrenetPlannerState.msg"
  "msg/FrenetTrajectoryPoint.msg"
  "msg/ReactiveAdvice.msg"
  DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
