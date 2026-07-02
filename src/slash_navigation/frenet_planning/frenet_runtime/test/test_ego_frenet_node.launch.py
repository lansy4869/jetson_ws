import math
import os
import time
import unittest

from ament_index_python.packages import get_package_share_directory
import launch
import launch_ros.actions
import launch_testing.actions
import pytest
import rclpy
from frenet_interfaces.msg import FrenetEgoState
from nav_msgs.msg import Odometry


@pytest.mark.launch_test
def generate_test_description():
    csv_path = os.path.join(
        get_package_share_directory("frenet_runtime"),
        "test",
        "ego_reference.csv",
    )

    ego_frenet_node = launch_ros.actions.Node(
        package="frenet_runtime",
        executable="ego_frenet_node",
        name="ego_frenet_node_test",
        output="screen",
        parameters=[
            {
                "centerline_csv": csv_path,
                "odom_topic": "/test/pf/pose/odom",
                "ego_state_topic": "/test/frenet/ego_state",
                "global_frame": "map",
            }
        ],
    )

    return (
        launch.LaunchDescription(
            [
                ego_frenet_node,
                launch_testing.actions.ReadyToTest(),
            ]
        ),
        {"ego_frenet_node": ego_frenet_node},
    )


class TestEgoFrenetNodeSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node("ego_frenet_node_smoke_test")

    def tearDown(self):
        self.node.destroy_node()

    def wait_for_subscription(self, publisher):
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and publisher.get_subscription_count() == 0:
            rclpy.spin_once(self.node, timeout_sec=0.1)

        self.assertGreater(publisher.get_subscription_count(), 0)

    def publish_until_next_state(self, publisher, odom_msg, received_states):
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            publisher.publish(odom_msg)
            rclpy.spin_once(self.node, timeout_sec=0.1)
            for state in reversed(received_states):
                if (
                    state.header.frame_id == odom_msg.header.frame_id
                    and state.header.stamp.sec == odom_msg.header.stamp.sec
                    and state.header.stamp.nanosec == odom_msg.header.stamp.nanosec
                ):
                    return state

        self.fail("matching ego state was not published")

    def assert_no_drive_publisher(self):
        publishers = self.node.get_publisher_names_and_types_by_node(
            "ego_frenet_node_test",
            "/",
        )
        published_topics = {topic: types for topic, types in publishers}
        self.assertIn("/test/frenet/ego_state", published_topics)
        self.assertNotIn("/drive", published_topics)
        published_types = [type_name for types in published_topics.values() for type_name in types]
        self.assertFalse(
            any("ackermann" in type_name.lower() for type_name in published_types),
            published_types,
        )

    def test_publishes_valid_reinitialized_state(self):
        received_states = []
        self.node.create_subscription(
            FrenetEgoState,
            "/test/frenet/ego_state",
            lambda msg: received_states.append(msg),
            10,
        )
        odom_publisher = self.node.create_publisher(Odometry, "/test/pf/pose/odom", 10)

        self.wait_for_subscription(odom_publisher)

        odom_msg = Odometry()
        odom_msg.header.frame_id = "/map"
        odom_msg.header.stamp.sec = 123
        odom_msg.header.stamp.nanosec = 456
        odom_msg.pose.pose.position.x = 4.0
        odom_msg.pose.pose.position.y = 0.0
        odom_msg.pose.pose.orientation.z = math.sin(0.25 * math.pi)
        odom_msg.pose.pose.orientation.w = math.cos(0.25 * math.pi)
        odom_msg.twist.twist.linear.x = 1.5

        state = self.publish_until_next_state(odom_publisher, odom_msg, received_states)

        self.assertEqual(state.header.frame_id, odom_msg.header.frame_id)
        self.assertEqual(state.header.stamp.sec, odom_msg.header.stamp.sec)
        self.assertEqual(state.header.stamp.nanosec, odom_msg.header.stamp.nanosec)
        self.assertTrue(state.valid)
        self.assertTrue(state.reinitialized)
        self.assertAlmostEqual(state.s, 0.0, places=6)
        self.assertAlmostEqual(state.s_unwrapped, 0.0, places=6)
        self.assertAlmostEqual(state.d, 0.0, places=6)
        self.assertAlmostEqual(state.yaw_error, 0.0, places=6)
        self.assertAlmostEqual(state.speed, 1.5, places=6)
        self.assert_no_drive_publisher()

        zero_quaternion_msg = Odometry()
        zero_quaternion_msg.header.frame_id = "map"
        zero_quaternion_msg.header.stamp.sec = 124
        zero_quaternion_msg.pose.pose.position.x = 4.0
        zero_quaternion_msg.pose.pose.position.y = 0.0
        zero_quaternion_msg.pose.pose.orientation.w = 0.0
        zero_quaternion_msg.twist.twist.linear.x = 1.5

        invalid_zero_quaternion = self.publish_until_next_state(
            odom_publisher,
            zero_quaternion_msg,
            received_states,
        )
        self.assertEqual(invalid_zero_quaternion.header.frame_id, "map")
        self.assertFalse(invalid_zero_quaternion.valid)
        self.assertFalse(invalid_zero_quaternion.reinitialized)
        self.assertAlmostEqual(invalid_zero_quaternion.s, 0.0, places=12)
        self.assertAlmostEqual(invalid_zero_quaternion.s_unwrapped, 0.0, places=12)
        self.assertAlmostEqual(invalid_zero_quaternion.d, 0.0, places=12)
        self.assertAlmostEqual(invalid_zero_quaternion.yaw_error, 0.0, places=12)
        self.assertAlmostEqual(invalid_zero_quaternion.speed, 0.0, places=12)

        bad_frame_msg = Odometry()
        bad_frame_msg.header.frame_id = "odom"
        bad_frame_msg.header.stamp.sec = 125
        bad_frame_msg.pose.pose.position.x = 4.0
        bad_frame_msg.pose.pose.position.y = 0.0
        bad_frame_msg.pose.pose.orientation.z = math.sin(0.25 * math.pi)
        bad_frame_msg.pose.pose.orientation.w = math.cos(0.25 * math.pi)
        bad_frame_msg.twist.twist.linear.x = 1.5

        invalid_bad_frame = self.publish_until_next_state(
            odom_publisher,
            bad_frame_msg,
            received_states,
        )
        self.assertEqual(invalid_bad_frame.header.frame_id, "odom")
        self.assertFalse(invalid_bad_frame.valid)
        self.assertFalse(invalid_bad_frame.reinitialized)
        self.assertAlmostEqual(invalid_bad_frame.s, 0.0, places=12)
        self.assertAlmostEqual(invalid_bad_frame.speed, 0.0, places=12)
