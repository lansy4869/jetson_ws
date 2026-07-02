cmake_minimum_required(VERSION 3.5)
project(frenet_local_planner)

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
find_package(builtin_interfaces REQUIRED)
find_package(frenet_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(track_spline REQUIRED)

add_library(frenet_local_planner_core
  src/state_machine.cpp
  src/trajectory_generator.cpp
)

target_compile_features(frenet_local_planner_core PUBLIC cxx_std_14)

target_include_directories(frenet_local_planner_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(frenet_local_planner_core
  builtin_interfaces
  frenet_interfaces
  std_msgs
  track_spline
)

add_executable(frenet_local_planner_node
  src/frenet_local_planner_node.cpp
)

target_compile_features(frenet_local_planner_node PUBLIC cxx_std_14)
target_link_libraries(frenet_local_planner_node frenet_local_planner_core)

ament_target_dependencies(frenet_local_planner_node
  ament_index_cpp
  frenet_interfaces
  geometry_msgs
  nav_msgs
  rclcpp
  std_msgs
  track_spline
)

install(TARGETS
  frenet_local_planner_core
  frenet_local_planner_node
  EXPORT export_frenet_local_planner_core
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

ament_export_targets(export_frenet_local_planner_core HAS_LIBRARY_TARGET)
ament_export_dependencies(
  builtin_interfaces
  frenet_interfaces
  std_msgs
  track_spline
)
ament_export_include_directories(include)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_state_machine
    test/test_state_machine.cpp
  )
  target_link_libraries(test_state_machine frenet_local_planner_core)

  ament_add_gtest(test_trajectory_generator
    test/test_trajectory_generator.cpp
  )
  target_link_libraries(test_trajectory_generator frenet_local_planner_core)
endif()

ament_package()
