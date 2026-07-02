cmake_minimum_required(VERSION 3.5)
project(frenet_runtime)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(frenet_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(track_spline REQUIRED)

add_library(ego_frenet_projector
  src/ego_frenet_projector.cpp
)

target_compile_features(ego_frenet_projector PUBLIC cxx_std_14)

target_include_directories(ego_frenet_projector PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(ego_frenet_projector track_spline)

add_executable(ego_frenet_node
  src/ego_frenet_node.cpp
)

target_compile_features(ego_frenet_node PUBLIC cxx_std_14)

target_link_libraries(ego_frenet_node ego_frenet_projector)

ament_target_dependencies(ego_frenet_node
  ament_index_cpp
  frenet_interfaces
  geometry_msgs
  nav_msgs
  rcl_interfaces
  rclcpp
  std_msgs
  tf2
  tf2_geometry_msgs
  track_spline
)

install(TARGETS
  ego_frenet_projector
  ego_frenet_node
  EXPORT export_ego_frenet_projector
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY
  config
  launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY test/data/
  DESTINATION share/${PROJECT_NAME}/test
)

ament_export_targets(export_ego_frenet_projector HAS_LIBRARY_TARGET)
ament_export_dependencies(
  ament_index_cpp
  frenet_interfaces
  geometry_msgs
  nav_msgs
  rcl_interfaces
  rclcpp
  std_msgs
  tf2
  tf2_geometry_msgs
  track_spline
)
ament_export_include_directories(include)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(launch_testing_ament_cmake REQUIRED)

  ament_add_gtest(test_ego_frenet_projector
    test/test_ego_frenet_projector.cpp
  )
  target_link_libraries(test_ego_frenet_projector ego_frenet_projector)
  add_launch_test(test/test_ego_frenet_node.launch.py TIMEOUT 30)
endif()

ament_package()
