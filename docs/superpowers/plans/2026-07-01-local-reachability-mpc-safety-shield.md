# Local Reachability MPC Safety Shield Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a real local-reachability supervised MPC Safety Shield by publishing Battle Fast2 reachability diagnostics, using those diagnostics in `drive_arbiter`, and updating the paper to match the implemented system.

**Architecture:** Battle Fast2 remains the default direct-drive baseline, but in the experimental launch it also publishes reachability diagnostics. `drive_arbiter_core.py` keeps its ROS-free pure-function design and adds optional Shield inputs; without fresh Shield diagnostics it preserves the current MPC-preferred fallback behavior. The paper is updated after code tests pass so the written method matches the actual implementation.

**Tech Stack:** ROS2 `rclpy`, `ackermann_msgs/AckermannDriveStamped`, `std_msgs/Float32`, Python dataclasses, pytest, existing LaTeX paper files.

## Global Constraints

- Do not change the default `battle_fast2.launch.py` behavior: `/scan -> battle_fast2_node -> /drive` must still work.
- Do not let multiple nodes publish `/drive` in `global_mpc_battle_arbiter.launch.py`.
- Do not modify or depend on the untracked `src/slash_navigation/frenet_planning/` directory.
- Do not introduce custom ROS messages; use `std_msgs/Float32` for first-version diagnostics.
- Preserve current fallback behavior when Shield diagnostics are missing or stale.
- Use TDD for production-code behavior changes: write a failing test, verify RED, implement, verify GREEN.

---

## File Structure

- Modify `src/slash_navigation/roboracer_china_2025/roboracer_china_2025/drive_arbiter_core.py`
  - Add `SafetyDiagnostics` and `ShieldConfig`.
  - Extend `arbitrate()` with optional Shield diagnostics while preserving existing behavior.
- Modify `src/slash_navigation/roboracer_china_2025/test/test_drive_arbiter_core.py`
  - Add tests for GREEN/YELLOW/ORANGE/RED/BLACK, stale diagnostics, and non-finite diagnostics.
- Modify `src/slash_navigation/roboracer_china_2025/roboracer_china_2025/battle_fast2_node.py`
  - Publish reachability diagnostics from `ReachabilityResult`.
- Modify `src/slash_navigation/roboracer_china_2025/package.xml`
  - Add `std_msgs` runtime dependency.
- Modify `src/slash_navigation/roboracer_china_2025/launch/global_mpc_battle_arbiter.launch.py`
  - Wire diagnostic topics and Shield thresholds into Battle Fast2 and Drive Arbiter.
- Modify `src/slash_navigation/roboracer_china_2025/launch/battle_fast2.launch.py`
  - Expose diagnostic topic parameters without changing default direct-drive behavior.
- Modify `src/slash_navigation/roboracer_china_2025/test/test_global_mpc_battle_launch.py`
  - Assert launch contains Shield topic and parameter names.
- Modify `paper/README.md`, `paper/main.tex`, `paper/chapters/abstract.tex`, `chapter01`, `chapter03`, `chapter04`, `chapter06`, `chapter07`, `chapter08`
  - Reframe the paper as local reachability supervised MPC Safety Shield.

---

### Task 1: Add Shield Decision Logic To Arbiter Core

**Files:**
- Modify: `src/slash_navigation/roboracer_china_2025/test/test_drive_arbiter_core.py`
- Modify: `src/slash_navigation/roboracer_china_2025/roboracer_china_2025/drive_arbiter_core.py`

**Interfaces:**
- Consumes: existing `DriveCommand`, `SourceHealth`, `ArbiterConfig`, `arbitrate(...)`.
- Produces:
  - `SafetyDiagnostics(front_clearance: float, risk_min_margin: float, reactive_speed_limit: float)`
  - `ShieldConfig(enabled: bool, diag_timeout_s: float, yellow_margin_m: float, orange_margin_m: float, red_margin_m: float, black_clearance_m: float, orange_blend: float)`
  - Extended `arbitrate(..., safety_diagnostics: Optional[SafetyDiagnostics] = None, safety_health: Optional[SourceHealth] = None, shield_config: Optional[ShieldConfig] = None) -> ArbitrationDecision`

- [ ] **Step 1: Write RED tests for Shield states**

Append these imports and helper to `test_drive_arbiter_core.py`:

```python
from roboracer_china_2025.drive_arbiter_core import SafetyDiagnostics, ShieldConfig


def shield_config():
    return ShieldConfig(
        enabled=True,
        diag_timeout_s=0.2,
        yellow_margin_m=0.55,
        orange_margin_m=0.35,
        red_margin_m=0.18,
        black_clearance_m=0.30,
        orange_blend=0.5,
    )
```

Append these tests:

```python
def test_shield_green_passes_mpc_when_margin_is_high():
    decision = arbitrate(
        now_s=1.0,
        mpc_command=DriveCommand(0.1, 1.2),
        mpc_health=healthy(0.95),
        reactive_command=DriveCommand(-0.3, 0.5),
        reactive_health=healthy(0.95),
        odom_health=healthy(0.95),
        config=config(),
        safety_diagnostics=SafetyDiagnostics(2.0, 0.8, 1.5),
        safety_health=healthy(0.95),
        shield_config=shield_config(),
    )

    assert decision.source == "mpc"
    assert decision.command == DriveCommand(0.1, 1.2)
    assert "green" in decision.reason


def test_shield_yellow_limits_mpc_speed_and_keeps_steering():
    decision = arbitrate(
        now_s=1.0,
        mpc_command=DriveCommand(0.1, 1.4),
        mpc_health=healthy(0.95),
        reactive_command=DriveCommand(-0.3, 0.6),
        reactive_health=healthy(0.95),
        odom_health=healthy(0.95),
        config=config(),
        safety_diagnostics=SafetyDiagnostics(2.0, 0.5, 0.7),
        safety_health=healthy(0.95),
        shield_config=shield_config(),
    )

    assert decision.source == "shield_yellow"
    assert decision.command == DriveCommand(0.1, 0.7)


def test_shield_orange_blends_steering_and_uses_conservative_speed():
    decision = arbitrate(
        now_s=1.0,
        mpc_command=DriveCommand(0.2, 1.3),
        mpc_health=healthy(0.95),
        reactive_command=DriveCommand(-0.4, 0.6),
        reactive_health=healthy(0.95),
        odom_health=healthy(0.95),
        config=config(),
        safety_diagnostics=SafetyDiagnostics(2.0, 0.25, 0.8),
        safety_health=healthy(0.95),
        shield_config=shield_config(),
    )

    assert decision.source == "shield_orange"
    assert decision.command == DriveCommand(-0.1, 0.6)


def test_shield_red_uses_reactive_command():
    decision = arbitrate(
        now_s=1.0,
        mpc_command=DriveCommand(0.2, 1.3),
        mpc_health=healthy(0.95),
        reactive_command=DriveCommand(-0.4, 0.6),
        reactive_health=healthy(0.95),
        odom_health=healthy(0.95),
        config=config(),
        safety_diagnostics=SafetyDiagnostics(2.0, 0.1, 0.8),
        safety_health=healthy(0.95),
        shield_config=shield_config(),
    )

    assert decision.source == "shield_red"
    assert decision.command == DriveCommand(-0.4, 0.6)


def test_shield_black_stops_when_front_clearance_is_too_low():
    decision = arbitrate(
        now_s=1.0,
        mpc_command=DriveCommand(0.2, 1.3),
        mpc_health=healthy(0.95),
        reactive_command=DriveCommand(-0.4, 0.6),
        reactive_health=healthy(0.95),
        odom_health=healthy(0.95),
        config=config(),
        safety_diagnostics=SafetyDiagnostics(0.2, 0.8, 0.5),
        safety_health=healthy(0.95),
        shield_config=shield_config(),
    )

    assert decision.source == "stop"
    assert decision.command == DriveCommand(0.0, 0.0)
    assert "black" in decision.reason


def test_stale_shield_diagnostics_preserve_original_mpc_behavior():
    decision = arbitrate(
        now_s=1.0,
        mpc_command=DriveCommand(0.1, 1.2),
        mpc_health=healthy(0.95),
        reactive_command=DriveCommand(-0.3, 0.5),
        reactive_health=healthy(0.95),
        odom_health=healthy(0.95),
        config=config(),
        safety_diagnostics=SafetyDiagnostics(0.2, 0.1, 0.0),
        safety_health=healthy(0.0),
        shield_config=shield_config(),
    )

    assert decision.source == "mpc"
    assert decision.command == DriveCommand(0.1, 1.2)


def test_non_finite_shield_diagnostics_stop():
    decision = arbitrate(
        now_s=1.0,
        mpc_command=DriveCommand(0.1, 1.2),
        mpc_health=healthy(0.95),
        reactive_command=DriveCommand(-0.3, 0.5),
        reactive_health=healthy(0.95),
        odom_health=healthy(0.95),
        config=config(),
        safety_diagnostics=SafetyDiagnostics(math.nan, 0.8, 0.5),
        safety_health=healthy(0.95),
        shield_config=shield_config(),
    )

    assert decision.source == "stop"
    assert "diagnostics" in decision.reason
```

- [ ] **Step 2: Run RED tests**

Run:

```bash
cd /home/wjh/下载/studies/slash_ws
PYTHONPATH=src/slash_navigation/roboracer_china_2025 pytest src/slash_navigation/roboracer_china_2025/test/test_drive_arbiter_core.py -q
```

Expected: FAIL with import error for `SafetyDiagnostics` or `ShieldConfig`.

- [ ] **Step 3: Implement minimal Shield logic**

In `drive_arbiter_core.py`, add:

```python
@dataclass(frozen=True)
class SafetyDiagnostics:
    front_clearance: float
    risk_min_margin: float
    reactive_speed_limit: float


@dataclass(frozen=True)
class ShieldConfig:
    enabled: bool
    diag_timeout_s: float
    yellow_margin_m: float
    orange_margin_m: float
    red_margin_m: float
    black_clearance_m: float
    orange_blend: float
```

Add helper functions:

```python
def _is_diagnostics_finite(diagnostics: Optional[SafetyDiagnostics]) -> bool:
    if diagnostics is None:
        return False
    return (
        math.isfinite(diagnostics.front_clearance)
        and math.isfinite(diagnostics.risk_min_margin)
        and math.isfinite(diagnostics.reactive_speed_limit)
    )


def _shield_active(
    now_s: float,
    safety_health: Optional[SourceHealth],
    shield_config: Optional[ShieldConfig],
) -> bool:
    if shield_config is None or not shield_config.enabled:
        return False
    if safety_health is None:
        return False
    return _is_fresh(now_s, safety_health, shield_config.diag_timeout_s)


def _risk_state(diagnostics: SafetyDiagnostics, shield_config: ShieldConfig) -> str:
    if not _is_diagnostics_finite(diagnostics):
        return "black"
    if diagnostics.front_clearance <= shield_config.black_clearance_m:
        return "black"
    if diagnostics.risk_min_margin <= shield_config.red_margin_m:
        return "red"
    if diagnostics.risk_min_margin <= shield_config.orange_margin_m:
        return "orange"
    if diagnostics.risk_min_margin <= shield_config.yellow_margin_m:
        return "yellow"
    return "green"
```

Then extend `arbitrate()` signature with optional Shield arguments and insert Shield handling after odom check and before the original MPC selection:

```python
def arbitrate(
    now_s: float,
    mpc_command: Optional[DriveCommand],
    mpc_health: SourceHealth,
    reactive_command: Optional[DriveCommand],
    reactive_health: SourceHealth,
    odom_health: SourceHealth,
    config: ArbiterConfig,
    safety_diagnostics: Optional[SafetyDiagnostics] = None,
    safety_health: Optional[SourceHealth] = None,
    shield_config: Optional[ShieldConfig] = None,
) -> ArbitrationDecision:
    ...
```

Use this behavior:

```python
    shield_is_active = _shield_active(now_s, safety_health, shield_config)
    mpc_fresh = _is_fresh(now_s, mpc_health, config.command_timeout_s)
    reactive_fresh = _is_fresh(now_s, reactive_health, config.command_timeout_s)

    if shield_is_active:
        if not _is_diagnostics_finite(safety_diagnostics):
            return _stop("shield diagnostics non-finite")
        state = _risk_state(safety_diagnostics, shield_config)
        if state == "black":
            return _stop("shield black")
        if mpc_fresh and _is_command_finite(mpc_command):
            if state == "green":
                if mpc_command.speed > safety_diagnostics.reactive_speed_limit:
                    limited = DriveCommand(
                        mpc_command.steering_angle,
                        safety_diagnostics.reactive_speed_limit,
                    )
                    return ArbitrationDecision("shield_yellow", _clamp_command(limited, config), "shield green speed limited")
                return ArbitrationDecision("mpc", _clamp_command(mpc_command, config), "shield green")
            if state == "yellow":
                limited = DriveCommand(
                    mpc_command.steering_angle,
                    min(mpc_command.speed, safety_diagnostics.reactive_speed_limit),
                )
                return ArbitrationDecision("shield_yellow", _clamp_command(limited, config), "shield yellow speed limit")
            if state == "orange":
                if reactive_fresh and _is_command_finite(reactive_command):
                    w = _clamp(shield_config.orange_blend, 0.0, 1.0)
                    blended = DriveCommand(
                        (1.0 - w) * mpc_command.steering_angle + w * reactive_command.steering_angle,
                        min(mpc_command.speed, reactive_command.speed, safety_diagnostics.reactive_speed_limit),
                    )
                    return ArbitrationDecision("shield_orange", _clamp_command(blended, config), "shield orange blend")
                limited = DriveCommand(
                    mpc_command.steering_angle,
                    min(mpc_command.speed, safety_diagnostics.reactive_speed_limit),
                )
                return ArbitrationDecision("shield_yellow", _clamp_command(limited, config), "shield orange degraded to speed limit")
        if state == "red":
            if reactive_fresh and _is_command_finite(reactive_command):
                return ArbitrationDecision("shield_red", _clamp_command(reactive_command, config), "shield red reactive")
            return _stop("shield red reactive unavailable")
```

Ensure the original fallback block remains after the Shield block.

- [ ] **Step 4: Run GREEN tests**

Run:

```bash
cd /home/wjh/下载/studies/slash_ws
PYTHONPATH=src/slash_navigation/roboracer_china_2025 pytest src/slash_navigation/roboracer_china_2025/test/test_drive_arbiter_core.py -q
```

Expected: all tests in that file pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/slash_navigation/roboracer_china_2025/roboracer_china_2025/drive_arbiter_core.py src/slash_navigation/roboracer_china_2025/test/test_drive_arbiter_core.py
git commit -m "feat: add local reachability shield core"
```

---

### Task 2: Publish Battle Fast2 Reachability Diagnostics

**Files:**
- Modify: `src/slash_navigation/roboracer_china_2025/roboracer_china_2025/battle_fast2_node.py`
- Modify: `src/slash_navigation/roboracer_china_2025/package.xml`
- Modify: `src/slash_navigation/roboracer_china_2025/test/test_battle_fast2_defaults.py`

**Interfaces:**
- Consumes: `ReachabilityResult.free_distance`, `ReachabilityResult.min_clearance`, `ReachabilityResult.speed`.
- Produces:
  - `/battle_fast2/front_clearance_m`
  - `/battle_fast2/risk_min_margin_m`
  - `/battle_fast2/reactive_speed_limit_mps`

- [ ] **Step 1: Write RED static tests**

Append to `test_battle_fast2_defaults.py`:

```python
def test_battle_fast2_publishes_reachability_shield_diagnostics():
    source = Path("roboracer_china_2025/battle_fast2_node.py").read_text()

    assert "from std_msgs.msg import Float32" in source
    assert "front_clearance_topic" in source
    assert "risk_min_margin_topic" in source
    assert "reactive_speed_limit_topic" in source
    assert "front_clearance_pub" in source
    assert "risk_min_margin_pub" in source
    assert "reactive_speed_limit_pub" in source
    assert "publish_reachability_diagnostics" in source


def test_package_declares_std_msgs_dependency():
    package_xml = Path("package.xml").read_text()
    assert "<exec_depend>std_msgs</exec_depend>" in package_xml
```

- [ ] **Step 2: Run RED tests**

Run:

```bash
cd /home/wjh/下载/studies/slash_ws/src/slash_navigation/roboracer_china_2025
pytest test/test_battle_fast2_defaults.py -q
```

Expected: FAIL because `std_msgs` import and diagnostic publisher names are missing.

- [ ] **Step 3: Implement diagnostic publishers**

In `battle_fast2_node.py`:

```python
from std_msgs.msg import Float32
```

Declare parameters:

```python
self.declare_parameter('front_clearance_topic', '/battle_fast2/front_clearance_m')
self.declare_parameter('risk_min_margin_topic', '/battle_fast2/risk_min_margin_m')
self.declare_parameter('reactive_speed_limit_topic', '/battle_fast2/reactive_speed_limit_mps')
```

Read parameters and create publishers:

```python
front_clearance_topic = self.get_parameter('front_clearance_topic').value
risk_min_margin_topic = self.get_parameter('risk_min_margin_topic').value
reactive_speed_limit_topic = self.get_parameter('reactive_speed_limit_topic').value

self.front_clearance_pub = self.create_publisher(Float32, front_clearance_topic, 10)
self.risk_min_margin_pub = self.create_publisher(Float32, risk_min_margin_topic, 10)
self.reactive_speed_limit_pub = self.create_publisher(Float32, reactive_speed_limit_topic, 10)
```

Add:

```python
def publish_reachability_diagnostics(self, front_clearance, risk_min_margin, reactive_speed_limit):
    front_msg = Float32()
    front_msg.data = float(front_clearance) if math.isfinite(float(front_clearance)) else 0.0
    self.front_clearance_pub.publish(front_msg)

    margin_msg = Float32()
    margin_msg.data = float(risk_min_margin) if math.isfinite(float(risk_min_margin)) else -1.0
    self.risk_min_margin_pub.publish(margin_msg)

    speed_msg = Float32()
    speed_msg.data = float(reactive_speed_limit) if math.isfinite(float(reactive_speed_limit)) else 0.0
    self.reactive_speed_limit_pub.publish(speed_msg)
```

Call it in `try_reachability_control()`:

```python
self.publish_reachability_diagnostics(0.0, -1.0, 0.0)
```

for invalid/error stop paths, and:

```python
self.publish_reachability_diagnostics(
    result.free_distance,
    result.min_clearance,
    result.speed,
)
```

before publishing valid reachability drive.

In `package.xml`, add:

```xml
<exec_depend>std_msgs</exec_depend>
```

- [ ] **Step 4: Run GREEN tests**

Run:

```bash
cd /home/wjh/下载/studies/slash_ws/src/slash_navigation/roboracer_china_2025
pytest test/test_battle_fast2_defaults.py -q
```

Expected: pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/slash_navigation/roboracer_china_2025/roboracer_china_2025/battle_fast2_node.py src/slash_navigation/roboracer_china_2025/package.xml src/slash_navigation/roboracer_china_2025/test/test_battle_fast2_defaults.py
git commit -m "feat: publish battle fast2 shield diagnostics"
```

---

### Task 3: Wire Shield Diagnostics Into Drive Arbiter Node And Launch

**Files:**
- Modify: `src/slash_navigation/roboracer_china_2025/roboracer_china_2025/drive_arbiter_node.py`
- Modify: `src/slash_navigation/roboracer_china_2025/launch/global_mpc_battle_arbiter.launch.py`
- Modify: `src/slash_navigation/roboracer_china_2025/launch/battle_fast2.launch.py`
- Modify: `src/slash_navigation/roboracer_china_2025/test/test_global_mpc_battle_launch.py`

**Interfaces:**
- Consumes: `SafetyDiagnostics`, `ShieldConfig`, three `Float32` diagnostic topics.
- Produces: `drive_arbiter` calls `arbitrate()` with Shield inputs in experimental launch.

- [ ] **Step 1: Write RED launch/static tests**

Append to `test_global_mpc_battle_launch.py`:

```python
def test_global_mpc_battle_launch_wires_safety_shield_topics():
    launch_text = Path("launch/global_mpc_battle_arbiter.launch.py").read_text()

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
    source = Path("roboracer_china_2025/drive_arbiter_node.py").read_text()

    assert "from std_msgs.msg import Float32" in source
    assert "SafetyDiagnostics" in source
    assert "ShieldConfig" in source
    assert "_front_clearance_callback" in source
    assert "_risk_min_margin_callback" in source
    assert "_reactive_speed_limit_callback" in source
```

- [ ] **Step 2: Run RED tests**

Run:

```bash
cd /home/wjh/下载/studies/slash_ws/src/slash_navigation/roboracer_china_2025
pytest test/test_global_mpc_battle_launch.py -q
```

Expected: FAIL because Shield topic constants and node subscriptions are missing.

- [ ] **Step 3: Implement node subscriptions**

In `drive_arbiter_node.py`, import:

```python
from std_msgs.msg import Float32
```

Import new core types:

```python
SafetyDiagnostics,
ShieldConfig,
```

Declare and read parameters:

```python
self.declare_parameter("enable_safety_shield", True)
self.declare_parameter("front_clearance_topic", "/battle_fast2/front_clearance_m")
self.declare_parameter("risk_min_margin_topic", "/battle_fast2/risk_min_margin_m")
self.declare_parameter("reactive_speed_limit_topic", "/battle_fast2/reactive_speed_limit_mps")
self.declare_parameter("shield_diag_timeout_s", 0.2)
self.declare_parameter("shield_yellow_margin_m", 0.55)
self.declare_parameter("shield_orange_margin_m", 0.35)
self.declare_parameter("shield_red_margin_m", 0.18)
self.declare_parameter("shield_black_clearance_m", 0.30)
self.declare_parameter("shield_orange_blend", 0.55)
```

Create `self.shield_config = ShieldConfig(...)`, `self.safety_diagnostics = None`, and `self.safety_health = SourceHealth(stamp_s=None, valid=False, reason="safety diagnostics missing")`.

Track three latest values:

```python
self.front_clearance = None
self.risk_min_margin = None
self.reactive_speed_limit = None
self.front_clearance_stamp_s = None
self.risk_min_margin_stamp_s = None
self.reactive_speed_limit_stamp_s = None
```

Add three subscriptions and callbacks. Each callback updates its value/stamp and calls:

```python
def _refresh_safety_diagnostics(self):
    if (
        self.front_clearance is None
        or self.risk_min_margin is None
        or self.reactive_speed_limit is None
    ):
        self.safety_health = SourceHealth(stamp_s=None, valid=False, reason="safety diagnostics missing")
        return
    latest = min(
        self.front_clearance_stamp_s,
        self.risk_min_margin_stamp_s,
        self.reactive_speed_limit_stamp_s,
    )
    self.safety_diagnostics = SafetyDiagnostics(
        front_clearance=float(self.front_clearance),
        risk_min_margin=float(self.risk_min_margin),
        reactive_speed_limit=float(self.reactive_speed_limit),
    )
    self.safety_health = SourceHealth(stamp_s=latest, valid=True)
```

Pass `safety_diagnostics`, `safety_health`, and `shield_config` into `arbitrate()`.

- [ ] **Step 4: Implement launch wiring**

In `global_mpc_battle_arbiter.launch.py`, add constants:

```python
FRONT_CLEARANCE_TOPIC = "/battle_fast2/front_clearance_m"
RISK_MIN_MARGIN_TOPIC = "/battle_fast2/risk_min_margin_m"
REACTIVE_SPEED_LIMIT_TOPIC = "/battle_fast2/reactive_speed_limit_mps"
```

Pass these topic parameters to both Battle Fast2 and Drive Arbiter. Declare Shield launch arguments:

```python
DeclareLaunchArgument("enable_safety_shield", default_value="true")
DeclareLaunchArgument("shield_diag_timeout_s", default_value="0.2")
DeclareLaunchArgument("shield_yellow_margin_m", default_value="0.55")
DeclareLaunchArgument("shield_orange_margin_m", default_value="0.35")
DeclareLaunchArgument("shield_red_margin_m", default_value="0.18")
DeclareLaunchArgument("shield_black_clearance_m", default_value="0.30")
DeclareLaunchArgument("shield_orange_blend", default_value="0.55")
```

In `battle_fast2.launch.py`, expose the three diagnostic topic arguments and pass them into the node without changing `drive_topic`.

- [ ] **Step 5: Run GREEN tests**

Run:

```bash
cd /home/wjh/下载/studies/slash_ws/src/slash_navigation/roboracer_china_2025
pytest test/test_global_mpc_battle_launch.py -q
```

Expected: pass.

- [ ] **Step 6: Commit Task 3**

```bash
git add src/slash_navigation/roboracer_china_2025/roboracer_china_2025/drive_arbiter_node.py src/slash_navigation/roboracer_china_2025/launch/global_mpc_battle_arbiter.launch.py src/slash_navigation/roboracer_china_2025/launch/battle_fast2.launch.py src/slash_navigation/roboracer_china_2025/test/test_global_mpc_battle_launch.py
git commit -m "feat: wire safety shield diagnostics into arbiter"
```

---

### Task 4: Update Paper To Safety Shield Mainline

**Files:**
- Modify: `../paper/README.md`
- Modify: `../paper/main.tex`
- Modify: `../paper/chapters/abstract.tex`
- Modify: `../paper/chapters/chapter01_introduction.tex`
- Modify: `../paper/chapters/chapter03_risk_perception_inputs.tex`
- Modify: `../paper/chapters/chapter04_battle_fast2_control.tex`
- Modify: `../paper/chapters/chapter06_drive_arbiter_safety.tex`
- Modify: `../paper/chapters/chapter07_experiments.tex`
- Modify: `../paper/chapters/chapter08_conclusion.tex`

**Interfaces:**
- Consumes: implemented code topic contract.
- Produces: paper text that describes local reachability supervised MPC Safety Shield instead of simple fallback arbitration.

- [ ] **Step 1: Write RED text consistency check**

Run:

```bash
cd /home/wjh/下载/studies
rg -n "局部可达性监督|Safety Shield|安全屏蔽|reactive_speed_limit|front_clearance_m|risk_min_margin_m" paper --glob '*.tex' --glob '*.md'
```

Expected before editing: too few or no matches for the new Shield mainline.

- [ ] **Step 2: Update title and README**

Change paper title in `main.tex` and `README.md` to:

```text
面向 F1TENTH 实车竞速的局部可达性监督 MPC 安全屏蔽控制方法研究
```

Update README architecture to include:

```text
/scan -> battle_fast2_node -> /battle_fast2/drive_reactive
                         -> /battle_fast2/front_clearance_m
                         -> /battle_fast2/risk_min_margin_m
                         -> /battle_fast2/reactive_speed_limit_mps
```

- [ ] **Step 3: Update chapter text**

Update the affected chapters with these required statements:

```text
Battle Fast2 不只是 fallback，而是局部可达性监督器。
Drive Arbiter 升级为 Safety Shield Arbiter。
GREEN/YELLOW/ORANGE/RED/BLACK 五档风险状态分别对应直通、限速、融合、接管和停车。
当前 Shield 使用 Battle Fast2 最优可达弧线的 free_distance、min_clearance 和 speed 作为第一版诊断量。
该方法不是严格 MPC 预测轨迹碰撞证明，而是局部可达性监督。
```

- [ ] **Step 4: Run GREEN text checks**

Run:

```bash
cd /home/wjh/下载/studies
rg -n "局部可达性监督|Safety Shield|安全屏蔽|reactive_speed_limit|front_clearance_m|risk_min_margin_m|GREEN|YELLOW|ORANGE|RED|BLACK" paper --glob '*.tex' --glob '*.md'
rg -n "只是 fallback|健康检查驱动的安全仲裁器|当前仲裁器适合定位为健康检查" paper --glob '*.tex' --glob '*.md'
```

Expected: first command has meaningful matches; second command has no stale mainline matches.

- [ ] **Step 5: Commit Task 4**

```bash
git add ../paper/README.md ../paper/main.tex ../paper/chapters/abstract.tex ../paper/chapters/chapter01_introduction.tex ../paper/chapters/chapter03_risk_perception_inputs.tex ../paper/chapters/chapter04_battle_fast2_control.tex ../paper/chapters/chapter06_drive_arbiter_safety.tex ../paper/chapters/chapter07_experiments.tex ../paper/chapters/chapter08_conclusion.tex
git commit -m "docs: update paper for safety shield mainline"
```

---

### Task 5: Final Verification

**Files:**
- No new source files.
- Verify all modified files.

**Interfaces:**
- Consumes all prior tasks.
- Produces final verified state.

- [ ] **Step 1: Run Python unit tests**

```bash
cd /home/wjh/下载/studies/slash_ws
PYTHONPATH=src/slash_navigation/roboracer_china_2025 pytest src/slash_navigation/roboracer_china_2025/test -q
```

Expected: pass, except any pre-existing external dependency issue must be recorded with exact output.

- [ ] **Step 2: Run Python syntax checks**

```bash
cd /home/wjh/下载/studies/slash_ws
python3 -m py_compile \
  src/slash_navigation/roboracer_china_2025/roboracer_china_2025/battle_fast2_node.py \
  src/slash_navigation/roboracer_china_2025/roboracer_china_2025/drive_arbiter_core.py \
  src/slash_navigation/roboracer_china_2025/roboracer_china_2025/drive_arbiter_node.py \
  src/slash_navigation/roboracer_china_2025/launch/global_mpc_battle_arbiter.launch.py \
  src/slash_navigation/roboracer_china_2025/launch/battle_fast2.launch.py
```

Expected: exit 0.

- [ ] **Step 3: Run paper consistency checks**

```bash
cd /home/wjh/下载/studies
rg -n "risk_mpc|sysid_ggv|reactive_advisor|R2-Gap|Corridor|local_corridor|健康检查驱动的安全仲裁器" paper --glob '*.tex' --glob '*.md'
rg -n "\\texttt\\{[^}]*_[^}]*\\}" paper --glob '*.tex' --glob '*.md'
```

Expected: no stale old-mainline matches and no unescaped underscore matches.

- [ ] **Step 4: Check git status**

```bash
cd /home/wjh/下载/studies/slash_ws
git status --short
```

Expected: only the known untracked `src/slash_navigation/frenet_planning/` remains if it was present before this work; no unintended files.

- [ ] **Step 5: Commit verification fixes if needed**

If any verification fix is needed:

```bash
git add <fixed-files>
git commit -m "fix: stabilize safety shield integration"
```

If no fixes are needed, do not create an empty commit.
