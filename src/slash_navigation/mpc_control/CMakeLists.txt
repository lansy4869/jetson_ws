cmake_minimum_required(VERSION 3.5)
project(mpc_control)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Enable user-local installs for OsqpEigen and custom Findosqp.cmake
list(APPEND CMAKE_MODULE_PATH "$ENV{HOME}/.local/share/cmake/Modules")
list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/.local")

find_package(Eigen3 REQUIRED)
find_package(OsqpEigen REQUIRED)

foreach(PACKAGE
  ament_cmake
  rclcpp
  rcl_interfaces
  ackermann_msgs
  ament_index_cpp
  geometry_msgs
  nav_msgs
  tf2
  tf2_geometry_msgs
  track_spline
  visualization_msgs)
  find_package(${PACKAGE} REQUIRED)
endforeach()

include_directories(include)
include_directories(${EIGEN3_INCLUDE_DIRS})

add_executable(mpc_node
  src/mpc_node.cpp
)

target_include_directories(mpc_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIRS}
)

ament_target_dependencies(mpc_node
  rclcpp
  rcl_interfaces
  ackermann_msgs
  ament_index_cpp
  geometry_msgs
  nav_msgs
  tf2
  tf2_geometry_msgs
  track_spline
  visualization_msgs
)

target_link_libraries(mpc_node OsqpEigen::OsqpEigen)

# Set RPATH so runtime linker finds OsqpEigen and OSQP shared libraries
set_target_properties(mpc_node PROPERTIES
  INSTALL_RPATH "$ENV{HOME}/.local/lib:/usr/local/lib/python3.8/dist-packages/casadi"
  BUILD_RPATH "$ENV{HOME}/.local/lib:/usr/local/lib/python3.8/dist-packages/casadi"
)

install(TARGETS mpc_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_mpc_utils
    test/test_mpc_utils.cpp
  )
  target_include_directories(test_mpc_utils PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${EIGEN3_INCLUDE_DIRS}
  )
  target_link_libraries(test_mpc_utils OsqpEigen::OsqpEigen)
endif()

ament_package()
