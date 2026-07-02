cmake_minimum_required(VERSION 3.5)
project(frenet_mpc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

list(APPEND CMAKE_MODULE_PATH "$ENV{HOME}/.local/share/cmake/Modules")
list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/.local")

find_package(Eigen3 REQUIRED)
find_package(OsqpEigen REQUIRED)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(ackermann_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(frenet_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

add_library(frenet_mpc_utils INTERFACE)
add_library(frenet_mpc::frenet_mpc_utils ALIAS frenet_mpc_utils)
target_compile_features(frenet_mpc_utils INTERFACE cxx_std_17)
target_include_directories(frenet_mpc_utils INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIRS}
)
target_link_libraries(frenet_mpc_utils INTERFACE
  OsqpEigen::OsqpEigen
)

add_library(frenet_mpc_adapter
  src/local_trajectory_adapter.cpp
)
target_compile_features(frenet_mpc_adapter PUBLIC cxx_std_17)
target_include_directories(frenet_mpc_adapter PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIRS}
)
ament_target_dependencies(frenet_mpc_adapter
  frenet_interfaces
)
target_link_libraries(frenet_mpc_adapter
  frenet_mpc_utils
)

add_executable(frenet_mpc_node
  src/frenet_mpc_node.cpp
)
target_compile_features(frenet_mpc_node PUBLIC cxx_std_17)
target_include_directories(frenet_mpc_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIRS}
)
ament_target_dependencies(frenet_mpc_node
  rclcpp
  rcl_interfaces
  ackermann_msgs
  builtin_interfaces
  frenet_interfaces
  geometry_msgs
  nav_msgs
  tf2
  tf2_geometry_msgs
)
target_link_libraries(frenet_mpc_node
  frenet_mpc_adapter
  OsqpEigen::OsqpEigen
)
set_target_properties(frenet_mpc_node PROPERTIES
  INSTALL_RPATH "$ENV{HOME}/.local/lib:/usr/local/lib/python3.8/dist-packages/casadi"
  BUILD_RPATH "$ENV{HOME}/.local/lib:/usr/local/lib/python3.8/dist-packages/casadi"
)
install(TARGETS frenet_mpc_node
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS
  frenet_mpc_utils
  EXPORT export_frenet_mpc
)

install(TARGETS
  frenet_mpc_adapter
  EXPORT export_frenet_mpc
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY include/
  DESTINATION include
  OPTIONAL
)

install(
  DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(Eigen3 OsqpEigen frenet_interfaces builtin_interfaces)
ament_export_include_directories(include)
ament_export_targets(export_frenet_mpc)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_mpc_utils
    test/test_mpc_utils.cpp
  )
  target_link_libraries(test_mpc_utils frenet_mpc_utils)

  ament_add_gtest(test_local_trajectory_adapter
    test/test_local_trajectory_adapter.cpp
  )
  target_link_libraries(test_local_trajectory_adapter frenet_mpc_adapter)
endif()

ament_package()
