import ast
from pathlib import Path
import xml.etree.ElementTree as ET


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_setup_exports_drive_arbiter_console_script():
    tree = ast.parse((PACKAGE_ROOT / "setup.py").read_text(encoding="utf-8"))
    scripts = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == "console_scripts":
                    scripts.extend(
                        item.value for item in value.elts if isinstance(item, ast.Constant)
                    )

    assert (
        "drive_arbiter = roboracer_china_2025.drive_arbiter_node:main"
        in scripts
    )


def test_package_declares_nav_msgs_dependency():
    root = ET.parse(PACKAGE_ROOT / "package.xml").getroot()
    deps = {element.text for element in root.findall("exec_depend")}

    assert "nav_msgs" in deps


def test_setup_installs_global_mpc_battle_launch():
    text = (PACKAGE_ROOT / "setup.py").read_text(encoding="utf-8")

    assert "launch/global_mpc_battle_arbiter.launch.py" in text


def test_experimental_launch_uses_private_control_topics():
    launch_text = (
        PACKAGE_ROOT / "launch" / "global_mpc_battle_arbiter.launch.py"
    ).read_text(encoding="utf-8")

    assert '"/mpc/drive_nominal"' in launch_text
    assert '"/battle_fast2/drive_reactive"' in launch_text
    assert '"drive_arbiter"' in launch_text
    assert '"battle_fast2_node"' in launch_text
    assert '"mpc_control"' in launch_text


def test_global_mpc_battle_launch_wires_safety_shield_topics():
    launch_text = (
        PACKAGE_ROOT / "launch" / "global_mpc_battle_arbiter.launch.py"
    ).read_text(encoding="utf-8")

    assert "FRONT_CLEARANCE_TOPIC" in launch_text
    assert "RISK_MIN_MARGIN_TOPIC" in launch_text
    assert "REACTIVE_SPEED_LIMIT_TOPIC" in launch_text
    assert "front_clearance_topic" in launch_text
    assert "risk_min_margin_topic" in launch_text
    assert "reactive_speed_limit_topic" in launch_text
    assert "enable_safety_shield" in launch_text
    assert "shield_yellow_margin_m" in launch_text
    assert "shield_orange_margin_m" in launch_text
    assert "shield_red_margin_m" in launch_text
    assert "shield_black_clearance_m" in launch_text


def test_drive_arbiter_node_subscribes_to_safety_shield_diagnostics():
    source = (PACKAGE_ROOT / "roboracer_china_2025/drive_arbiter_node.py").read_text(
        encoding="utf-8"
    )

    assert "from std_msgs.msg import Float32" in source
    assert "SafetyDiagnostics" in source
    assert "ShieldConfig" in source
    assert "_front_clearance_callback" in source
    assert "_risk_min_margin_callback" in source
    assert "_reactive_speed_limit_callback" in source
