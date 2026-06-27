from pathlib import Path

import yaml


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "joy_teleop.yaml"


def _joy_teleop_params():
    with CONFIG_PATH.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)["joy_teleop"]["ros__parameters"]


def test_default_command_does_not_publish_zero_teleop():
    params = _joy_teleop_params()

    assert params["default"]["topic_name"] != "teleop"


def test_human_control_still_publishes_teleop_override():
    params = _joy_teleop_params()

    assert params["human_control"]["topic_name"] == "teleop"
    assert params["human_control"]["deadman_buttons"] == [4]
