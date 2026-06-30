from pathlib import Path


NODE_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "roboracer_china_2025"
    / "battle_fast2_node.py"
)


def test_battle_fast2_node_uses_odometry_speed_for_reachability():
    source = NODE_SOURCE.read_text(encoding="utf-8")

    assert "from nav_msgs.msg import Odometry" in source
    assert "self.declare_parameter('odom_topic', '/odom')" in source
    assert "self.odom_sub = self.create_subscription(" in source
    assert "def odom_callback(self, msg: Odometry):" in source
    assert "current_speed=self.current_measured_speed()" in source
    assert "current_speed=float(self.last_reachability_speed)" not in source


def test_battle_fast2_node_wires_local_corridor_into_reachability():
    source = NODE_SOURCE.read_text(encoding="utf-8")

    assert "estimate_local_corridor" in source
    assert "self.previous_corridor" in source
    assert "corridor=corridor" in source
