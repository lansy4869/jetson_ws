import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _parse(relative_path):
    return ast.parse((PACKAGE_ROOT / relative_path).read_text(encoding="utf-8"))


def test_node_defaults_to_pure_reachability_without_original_fallback():
    tree = _parse("roboracer_china_2025/battle_fast2_node.py")

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "declare_parameter":
            continue
        if len(node.args) < 2:
            continue
        if not isinstance(node.args[0], ast.Constant):
            continue
        if node.args[0].value == "reachability_fallback_to_original":
            assert isinstance(node.args[1], ast.Constant)
            assert node.args[1].value is False
            return

    raise AssertionError("reachability_fallback_to_original parameter was not declared")


def test_launch_defaults_to_pure_reachability_without_original_fallback():
    tree = _parse("launch/battle_fast2.launch.py")

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "DeclareLaunchArgument":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        if node.args[0].value != "reachability_fallback_to_original":
            continue
        default_value = next(
            (keyword.value for keyword in node.keywords if keyword.arg == "default_value"),
            None,
        )
        assert isinstance(default_value, ast.Constant)
        assert default_value.value == "false"
        return

    raise AssertionError("reachability_fallback_to_original launch argument was not declared")


def test_battle_fast2_publishes_reachability_shield_diagnostics():
    source = (PACKAGE_ROOT / "roboracer_china_2025/battle_fast2_node.py").read_text(
        encoding="utf-8"
    )

    assert "from std_msgs.msg import Float32" in source
    assert "front_clearance_topic" in source
    assert "risk_min_margin_topic" in source
    assert "reactive_speed_limit_topic" in source
    assert "front_clearance_pub" in source
    assert "risk_min_margin_pub" in source
    assert "reactive_speed_limit_pub" in source
    assert "publish_reachability_diagnostics" in source


def test_package_declares_std_msgs_dependency():
    package_xml = (PACKAGE_ROOT / "package.xml").read_text(encoding="utf-8")
    assert "<exec_depend>std_msgs</exec_depend>" in package_xml
