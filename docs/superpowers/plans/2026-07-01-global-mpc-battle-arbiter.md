# Global MPC Battle Arbiter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an experimental global MPC + Battle Fast2 control mode where only a new arbiter publishes `/drive`.

**Architecture:** Keep the proven `battle_fast2.launch.py` unchanged. Add a pure Python arbitration core, a ROS 2 node wrapper, and a separate experimental launch that remaps MPC to `/mpc/drive_nominal`, Battle Fast2 to `/battle_fast2/drive_reactive`, and the arbiter to `/drive`.

**Tech Stack:** ROS 2 Python package `roboracer_china_2025`, `rclpy`, `ackermann_msgs`, `nav_msgs`, Python `pytest`, existing `mpc_control` launch/node contract.

## Global Constraints

- Do not change the default behavior of `launch/battle_fast2.launch.py`.
- Do not allow more than one node to directly publish `/drive` in the experimental launch.
- First implementation uses `mpc_control`, not full `frenet_planning`.
- `battle_fast2_node` remains a reactive safety source and publishes to `/battle_fast2/drive_reactive` in the experimental launch.
- `drive_arbiter` is the only experimental node that publishes `/drive`.
- Any non-finite command value must be rejected.
- Any stale odometry must make the arbiter output a stop command.
- Output steering and speed must be clamped before publish.

---

## File Structure

- Create `src/slash_navigation/roboracer_china_2025/roboracer_china_2025/drive_arbiter_core.py`
  - Pure, ROS-free arbitration logic for unit testing.
  - Defines command, health, config, and decision dataclasses.
  - Produces a selected source and clamped output command.

- Create `src/slash_navigation/roboracer_china_2025/roboracer_china_2025/drive_arbiter_node.py`
  - ROS 2 wrapper around `drive_arbiter_core`.
  - Subscribes to nominal MPC drive, reactive Battle Fast2 drive, and PF odom.
  - Publishes the final `/drive`.

- Create `src/slash_navigation/roboracer_china_2025/launch/global_mpc_battle_arbiter.launch.py`
  - Experimental launch only.
  - Starts `mpc_control`, `battle_fast2_node`, and `drive_arbiter`.

- Modify `src/slash_navigation/roboracer_china_2025/setup.py`
  - Install the new launch file.
  - Add `drive_arbiter` console entry point.

- Modify `src/slash_navigation/roboracer_china_2025/package.xml`
  - Add `nav_msgs` runtime dependency for odometry.

- Create `src/slash_navigation/roboracer_china_2025/test/test_drive_arbiter_core.py`
  - Unit tests for arbitration behavior.

- Create `src/slash_navigation/roboracer_china_2025/test/test_global_mpc_battle_launch.py`
  - Static launch/setup tests to prevent `/drive` topic regressions.

---

### Task 1: Pure Arbitration Core

**Files:**
- Create: `src/slash_navigation/roboracer_china_2025/roboracer_china_2025/drive_arbiter_core.py`
- Test: `src/slash_navigation/roboracer_china_2025/test/test_drive_arbiter_core.py`

**Interfaces:**
- Produces:
  - `DriveCommand(steering_angle: float, speed: float)`
  - `SourceHealth(stamp_s: Optional[float], valid: bool, reason: str = "")`
  - `ArbiterConfig(max_steer: float, max_speed: float, min_speed: float, command_timeout_s: float, odom_timeout_s: float, prefer_reactive_on_mpc_timeout: bool)`
  - `ArbitrationDecision(source: str, command: DriveCommand, reason: str)`
  - `arbitrate(now_s: float, mpc_command: Optional[DriveCommand], mpc_health: SourceHealth, reactive_command: Optional[DriveCommand], reactive_health: SourceHealth, odom_health: SourceHealth, config: ArbiterConfig) -> ArbitrationDecision`

- [ ] **Step 1: Write failing core tests**

Create `test/test_drive_arbiter_core.py` with tests for MPC selection, odom stale stop, reactive fallback on MPC timeout, non-finite rejection, and output clamping.

```python
import math

from roboracer_china_2025.drive_arbiter_core import (
    ArbiterConfig,
    DriveCommand,
    SourceHealth,
    arbitrate,
)


def config():
    return ArbiterConfig(
        max_steer=0.4,
        max_speed=2.0,
        min_speed=0.0,
        command_timeout_s=0.2,
        odom_timeout_s=0.3,
        prefer_reactive_on_mpc_timeout=True,
    )


def healthy(stamp_s=1.0):
    return SourceHealth(stamp_s=stamp_s, valid=True)


def test_selects_mpc_when_all_inputs_are_healthy():
    decision = arbitrate(
        now_s=1.1,
        mpc_command=DriveCommand(0.1, 1.5),
        mpc_health=healthy(1.05),
        reactive_command=DriveCommand(-0.2, 0.8),
        reactive_health=healthy(1.05),
        odom_health=healthy(1.05),
        config=config(),
    )

    assert decision.source == "mpc"
    assert decision.command == DriveCommand(0.1, 1.5)


def test_stops_when_odom_is_stale():
    decision = arbitrate(
        now_s=1.0,
        mpc_command=DriveCommand(0.1, 1.0),
        mpc_health=healthy(0.95),
        reactive_command=DriveCommand(-0.1, 0.5),
        reactive_health=healthy(0.95),
        odom_health=healthy(0.1),
        config=config(),
    )

    assert decision.source == "stop"
    assert decision.command == DriveCommand(0.0, 0.0)
    assert "odom stale" in decision.reason


def test_reactive_fallback_is_used_when_mpc_command_is_stale():
    decision = arbitrate(
        now_s=1.0,
        mpc_command=DriveCommand(0.1, 1.0),
        mpc_health=healthy(0.0),
        reactive_command=DriveCommand(-0.2, 0.7),
        reactive_health=healthy(0.95),
        odom_health=healthy(0.95),
        config=config(),
    )

    assert decision.source == "reactive"
    assert decision.command == DriveCommand(-0.2, 0.7)


def test_non_finite_mpc_command_falls_back_to_reactive():
    decision = arbitrate(
        now_s=1.0,
        mpc_command=DriveCommand(math.nan, 1.0),
        mpc_health=healthy(0.95),
        reactive_command=DriveCommand(-0.2, 0.7),
        reactive_health=healthy(0.95),
        odom_health=healthy(0.95),
        config=config(),
    )

    assert decision.source == "reactive"
    assert decision.command == DriveCommand(-0.2, 0.7)


def test_outputs_stop_when_both_commands_are_invalid():
    decision = arbitrate(
        now_s=1.0,
        mpc_command=DriveCommand(math.nan, 1.0),
        mpc_health=healthy(0.95),
        reactive_command=DriveCommand(0.1, math.inf),
        reactive_health=healthy(0.95),
        odom_health=healthy(0.95),
        config=config(),
    )

    assert decision.source == "stop"
    assert decision.command == DriveCommand(0.0, 0.0)


def test_selected_command_is_clamped():
    decision = arbitrate(
        now_s=1.0,
        mpc_command=DriveCommand(1.5, 9.0),
        mpc_health=healthy(0.95),
        reactive_command=None,
        reactive_health=SourceHealth(stamp_s=None, valid=False),
        odom_health=healthy(0.95),
        config=config(),
    )

    assert decision.source == "mpc"
    assert decision.command == DriveCommand(0.4, 2.0)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
cd /home/wjh/下载/studies/slash_ws/src/slash_navigation/roboracer_china_2025
python3 -m pytest test/test_drive_arbiter_core.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'roboracer_china_2025.drive_arbiter_core'`.

- [ ] **Step 3: Implement core**

Create `drive_arbiter_core.py` with the dataclasses, finite checks, stale checks, clamping, and selection logic required by the tests.

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```bash
cd /home/wjh/下载/studies/slash_ws/src/slash_navigation/roboracer_china_2025
python3 -m pytest test/test_drive_arbiter_core.py -q
```

Expected: `6 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/slash_navigation/roboracer_china_2025/roboracer_china_2025/drive_arbiter_core.py \
        src/slash_navigation/roboracer_china_2025/test/test_drive_arbiter_core.py
git commit -m "feat: add drive arbiter core"
```

---

### Task 2: ROS Drive Arbiter Node

**Files:**
- Create: `src/slash_navigation/roboracer_china_2025/roboracer_china_2025/drive_arbiter_node.py`
- Modify: `src/slash_navigation/roboracer_china_2025/setup.py`
- Modify: `src/slash_navigation/roboracer_china_2025/package.xml`
- Test: `src/slash_navigation/roboracer_china_2025/test/test_global_mpc_battle_launch.py`

**Interfaces:**
- Consumes: `arbitrate()` and dataclasses from Task 1.
- Produces:
  - Console script `drive_arbiter = roboracer_china_2025.drive_arbiter_node:main`
  - ROS parameters:
    - `mpc_drive_topic`, default `/mpc/drive_nominal`
    - `reactive_drive_topic`, default `/battle_fast2/drive_reactive`
    - `odom_topic`, default `/pf/pose/odom`
    - `drive_topic`, default `/drive`
    - `global_frame`, default `map`
    - `publish_rate_hz`, default `30.0`
    - `command_timeout_s`, default `0.2`
    - `odom_timeout_s`, default `0.3`
    - `max_steer`, default `0.412`
    - `max_speed`, default `2.0`
    - `min_speed`, default `0.0`

- [ ] **Step 1: Write failing static packaging test**

Create or extend `test/test_global_mpc_battle_launch.py` with AST assertions that `setup.py` exposes the `drive_arbiter` entry point and `package.xml` declares `nav_msgs`.

```python
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
```

- [ ] **Step 2: Run test and verify RED**

Run:

```bash
cd /home/wjh/下载/studies/slash_ws/src/slash_navigation/roboracer_china_2025
python3 -m pytest test/test_global_mpc_battle_launch.py -q
```

Expected: FAIL because `drive_arbiter` and `nav_msgs` are not declared yet.

- [ ] **Step 3: Implement ROS node and package wiring**

Create `drive_arbiter_node.py`. Update `setup.py` entry points and `package.xml`.

The node must convert incoming ROS messages to `DriveCommand` and `SourceHealth`, call `arbitrate()` on a timer, and publish `AckermannDriveStamped` to `drive_topic`.

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```bash
cd /home/wjh/下载/studies/slash_ws/src/slash_navigation/roboracer_china_2025
python3 -m pytest test/test_global_mpc_battle_launch.py test/test_drive_arbiter_core.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/slash_navigation/roboracer_china_2025/roboracer_china_2025/drive_arbiter_node.py \
        src/slash_navigation/roboracer_china_2025/setup.py \
        src/slash_navigation/roboracer_china_2025/package.xml \
        src/slash_navigation/roboracer_china_2025/test/test_global_mpc_battle_launch.py
git commit -m "feat: add drive arbiter ros node"
```

---

### Task 3: Experimental Launch

**Files:**
- Create: `src/slash_navigation/roboracer_china_2025/launch/global_mpc_battle_arbiter.launch.py`
- Modify: `src/slash_navigation/roboracer_china_2025/setup.py`
- Test: `src/slash_navigation/roboracer_china_2025/test/test_global_mpc_battle_launch.py`

**Interfaces:**
- Consumes: `drive_arbiter` console script from Task 2.
- Produces:
  - Launch file `global_mpc_battle_arbiter.launch.py`
  - Launch args: `scan_topic`, `odom_topic`, `waypoint_csv`, `mpc_params_file`, `trajectory_mode`, `drive_topic`, `max_speed`, `max_steer`
  - Topic contract:
    - MPC node publishes `/mpc/drive_nominal`
    - Battle Fast2 publishes `/battle_fast2/drive_reactive`
    - Arbiter publishes `/drive`

- [ ] **Step 1: Write failing launch tests**

Extend `test/test_global_mpc_battle_launch.py` with static checks that setup installs the new launch file and the launch source contains the required topic remaps.

```python
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
```

- [ ] **Step 2: Run test and verify RED**

Run:

```bash
cd /home/wjh/下载/studies/slash_ws/src/slash_navigation/roboracer_china_2025
python3 -m pytest test/test_global_mpc_battle_launch.py -q
```

Expected: FAIL because `global_mpc_battle_arbiter.launch.py` does not exist or is not installed.

- [ ] **Step 3: Implement launch file and install wiring**

Create `launch/global_mpc_battle_arbiter.launch.py` and update `setup.py` data files to install both launch files.

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```bash
cd /home/wjh/下载/studies/slash_ws/src/slash_navigation/roboracer_china_2025
python3 -m pytest test/test_global_mpc_battle_launch.py test/test_drive_arbiter_core.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/slash_navigation/roboracer_china_2025/launch/global_mpc_battle_arbiter.launch.py \
        src/slash_navigation/roboracer_china_2025/setup.py \
        src/slash_navigation/roboracer_china_2025/test/test_global_mpc_battle_launch.py
git commit -m "feat: add global mpc battle arbiter launch"
```

---

### Task 4: Verification and Documentation Notes

**Files:**
- Modify: `docs/superpowers/specs/2026-07-01-global-mpc-battle-fast2-arbiter-design.md`

**Interfaces:**
- Consumes: implemented launch and node names from Tasks 1-3.
- Produces: updated verification notes with exact commands.

- [ ] **Step 1: Update spec verification section**

Add the exact experimental launch command:

```bash
ros2 launch roboracer_china_2025 global_mpc_battle_arbiter.launch.py waypoint_csv:=/absolute/path/to/track.csv
```

Add the topic safety check:

```bash
ros2 topic info /drive
```

Expected on the robot: only `drive_arbiter` publishes `/drive`.

- [ ] **Step 2: Run final local tests**

Run:

```bash
cd /home/wjh/下载/studies/slash_ws/src/slash_navigation/roboracer_china_2025
python3 -m pytest test/test_drive_arbiter_core.py test/test_global_mpc_battle_launch.py test/test_battle_fast2_defaults.py -q
```

Expected: all selected tests pass.

- [ ] **Step 3: Run available workspace checks**

Run:

```bash
cd /home/wjh/下载/studies/slash_ws
python3 -m pytest src/slash_navigation/roboracer_china_2025/test/test_drive_arbiter_core.py \
                 src/slash_navigation/roboracer_china_2025/test/test_global_mpc_battle_launch.py \
                 src/slash_navigation/roboracer_china_2025/test/test_battle_fast2_defaults.py -q
```

Expected: all selected tests pass.

`colcon` is not available in this shell, so full ROS build verification must be run on the ROS2/Jetson environment.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-07-01-global-mpc-battle-fast2-arbiter-design.md
git commit -m "docs: document global mpc arbiter verification"
```
