# Gesture-Controlled Gazebo Car Worklog

This document explains how the SolidWorks car model, ROS 2 Foxy package, Gazebo simulation, and Windows hand gesture recognizer were connected into a working gesture-controlled car.

The final result is:

- The car model spawns in Gazebo.
- A Windows webcam gesture recognizer classifies hand gestures.
- The recognizer sends UDP JSON commands to WSL.
- A ROS 2 node in WSL receives those commands.
- The ROS 2 node directly controls the Gazebo model using Gazebo entity state services.

## Final Architecture

```text
Windows webcam
    |
    v
MediaPipe + trained gesture model
    |
    v
C:\Users\Yoav\Desktop\Personal Projects\hand-gestured-car\src\live_recognize.py
    |
    | UDP JSON on port 4210
    v
WSL Ubuntu-20.04 / ros_joe
    |
    v
/home/ros_joe/HandGestureDrone_ws/src/car_assembly_2nd/scripts/gesture_gazebo_driver
    |
    | /get_entity_state and /set_entity_state
    v
Gazebo model: car_assembly_2nd
```

V1 intentionally uses direct Gazebo model velocity control. It does not yet use wheel physics, a differential drive plugin, or ROS 2 control.

## Important Project Paths

Windows gesture project:

```text
C:\Users\Yoav\Desktop\Personal Projects\hand-gestured-car
```

Main gesture recognizer:

```text
C:\Users\Yoav\Desktop\Personal Projects\hand-gestured-car\src\live_recognize.py
```

Gesture model files:

```text
C:\Users\Yoav\Desktop\Personal Projects\hand-gestured-car\models\gesture_rf.joblib
C:\Users\Yoav\Desktop\Personal Projects\hand-gestured-car\models\labels.txt
```

ROS 2 workspace:

```text
/home/ros_joe/HandGestureDrone_ws
```

Active ROS 2 car package:

```text
/home/ros_joe/HandGestureDrone_ws/src/car_assembly_2nd
```

Active URDF:

```text
/home/ros_joe/HandGestureDrone_ws/src/car_assembly_2nd/urdf/car_assembly_2nd.urdf
```

Gazebo gesture driver:

```text
/home/ros_joe/HandGestureDrone_ws/src/car_assembly_2nd/scripts/gesture_gazebo_driver
```

Custom Gazebo world:

```text
/home/ros_joe/HandGestureDrone_ws/src/car_assembly_2nd/worlds/gesture_empty.world
```

## Gesture Recognition Side

The Windows project uses:

- OpenCV for webcam capture.
- MediaPipe Hands for hand landmarks.
- A trained model loaded from `models/gesture_rf.joblib`.
- Gesture labels from `models/labels.txt`.

The current gesture labels are:

```text
STOP_IDLE
SWIPE_LEFT
SWIPE_RIGHT
PUSH
PULL
CIRCLE_CW
CIRCLE_CCW
```

The recognizer uses a sliding time window of landmark motion and classifies the movement. It smooths predictions using:

- `STREAK_REQUIRED = 3`
- `CONF_THRESH = 0.70`
- `COOLDOWN_SECONDS = 0.6`

## Current Gesture Controls

The current control mapping is:

```text
CIRCLE_CW    -> arm
CIRCLE_CCW   -> arm
STOP_IDLE    -> disarm and STOP
PUSH         -> FORWARD
PULL         -> BACKWARD
SWIPE_LEFT   -> LEFT
SWIPE_RIGHT  -> RIGHT
```

Movement commands only work while armed.

This prevents random gestures from moving the car immediately. You first make a circle gesture to arm the bridge, then use movement gestures. Holding still or showing `STOP_IDLE` disarms the system and sends `STOP`.

The UDP payload sent from Windows looks like this:

```json
{
  "cmd": "FORWARD",
  "gesture": "PUSH",
  "confidence": 0.84,
  "hand": "Right",
  "ts": 1780000000.0
}
```

The recognizer sends a heartbeat every `0.2s` with the current latched command, so the ROS side keeps receiving the command while the gesture is active.

## ROS 2 Package Cleanup

The first SolidWorks export had a package/folder name like:

```text
Car_Assem_1st.SLDASM
```

That name is awkward for ROS packages because it contains uppercase letters and a dot. The mesh paths originally used:

```text
package://Car_Assem_1st.SLDASM/meshes/...
```

For ROS 2, the package was converted to a proper description package name:

```text
car_assem_1st_description
```

Then the active second model package became:

```text
car_assembly_2nd
```

The mesh paths now use a valid package name:

```text
package://car_assembly_2nd/meshes/base_link.STL
package://car_assembly_2nd/meshes/wheel_front_left.STL
package://car_assembly_2nd/meshes/wheel_front_right.STL
package://car_assembly_2nd/meshes/wheel_rear_left.STL
package://car_assembly_2nd/meshes/wheel_rear_right.STL
```

The ROS 2 package has:

```text
package.xml
CMakeLists.txt
urdf/
meshes/
textures/
launch/
config/
scripts/
worlds/
env-hooks/
```

## URDF And TF Fixes

The SolidWorks URDF caused TF errors like:

```text
TF NAN_INPUT
TF DENORMALIZED_QUATERNION
```

These errors usually mean `robot_state_publisher` tried to publish a transform with invalid numeric values. The common causes are:

- Invalid `origin xyz`.
- Invalid `origin rpy`.
- Empty numeric fields.
- `nan`, `-nan`, or `inf`.
- A continuous or revolute joint with an invalid zero axis like `axis xyz="0 0 0"`.
- Link and joint name collisions.

The wheel area was inspected carefully. The important wheel links were:

```text
wheel_front_left
wheel_front_right
wheel_rear_left
wheel_rear_right
```

The URDF was fixed so the wheel spin joints are valid continuous joints with a non-zero local Z axis:

```xml
<joint name="wheel_front_left_spin_joint" type="continuous">
  <parent link="motor_wheel_front_left" />
  <child link="wheel_front_left" />
  <axis xyz="0 0 1" />
</joint>
```

The same pattern was applied to:

```text
wheel_front_right_spin_joint
wheel_rear_left_spin_joint
wheel_rear_right_spin_joint
```

The motor mount joints are fixed joints:

```text
motor_wheel_front_left_mount_joint
motor_wheel_front_right_mount_joint
motor_wheel_rear_left_mount_joint
motor_wheel_rear_right_mount_joint
```

The joint names were changed so they do not collide with link names. For example:

```text
link:  wheel_front_left
joint: wheel_front_left_spin_joint
```

This matters because ROS, TF, Gazebo, and visualization tools become confusing or unstable when links and joints reuse the same names.

## RViz Visibility Fixes

RViz needed:

- A valid `/robot_description`.
- A valid TF tree.
- Mesh paths that resolve through ROS package lookup.

The expected RViz setup is:

```text
Fixed Frame: base_link
RobotModel Description Topic: /robot_description
```

The useful checks are:

```bash
cd ~/HandGestureDrone_ws
source /opt/ros/foxy/setup.bash
source install/setup.bash

check_urdf ~/HandGestureDrone_ws/src/car_assembly_2nd/urdf/car_assembly_2nd.urdf
ros2 pkg prefix car_assembly_2nd
```

## Gazebo Mesh Loading Fix

Gazebo was able to see that a model existed, but it could not load the visual mesh files.

The key concept is the installed package `share` directory. After `colcon build`, ROS installs package resources under:

```text
~/HandGestureDrone_ws/install/car_assembly_2nd/share/car_assembly_2nd
```

That installed `share` folder contains the package's URDF, meshes, textures, worlds, launch files, and other runtime assets.

When Gazebo sees paths like:

```text
model://...
package://...
```

it needs search paths that tell it where those package resources live.

An environment hook was added:

```text
/home/ros_joe/HandGestureDrone_ws/src/car_assembly_2nd/env-hooks/gazebo_model_path.dsv
```

It contains:

```text
prepend-non-duplicate;GAZEBO_MODEL_PATH;share
prepend-non-duplicate;GAZEBO_MODEL_PATH;/usr/share/gazebo-11/models
prepend-non-duplicate;GAZEBO_RESOURCE_PATH;/usr/share/gazebo-11
```

The package `CMakeLists.txt` registers the hook:

```cmake
ament_environment_hooks(
  env-hooks/gazebo_model_path.dsv
)
```

After this, when you run:

```bash
source ~/HandGestureDrone_ws/install/setup.bash
```

the Gazebo search path is updated automatically.

## Why We Added A Custom Gazebo World

The ROS gesture driver controls the model through these Gazebo services:

```text
/get_entity_state
/set_entity_state
```

Those services require the Gazebo ROS state plugin:

```xml
<plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so">
  <ros>
    <namespace>/</namespace>
  </ros>
</plugin>
```

So we created:

```text
/home/ros_joe/HandGestureDrone_ws/src/car_assembly_2nd/worlds/gesture_empty.world
```

That world includes:

- `libgazebo_ros_state.so`
- `ground_plane`
- `sun`

Without this world/plugin, the driver can run but cannot control the model because `/get_entity_state` and `/set_entity_state` may not exist.

## ROS Gesture Gazebo Driver

The driver is:

```text
/home/ros_joe/HandGestureDrone_ws/src/car_assembly_2nd/scripts/gesture_gazebo_driver
```

It is installed as a ROS executable, so it can run with:

```bash
ros2 run car_assembly_2nd gesture_gazebo_driver
```

The driver:

- Listens for UDP JSON on `0.0.0.0:4210`.
- Stores the latest valid command.
- Times out to `STOP` if no UDP packet arrives for more than `1.0s`.
- Calls `/get_entity_state` at 20 Hz.
- Preserves the current pose.
- Computes a twist command.
- Calls `/set_entity_state`.

The command speeds are:

```text
FORWARD:  +0.35 m/s
BACKWARD: -0.25 m/s
LEFT:     +1.8 rad/s
RIGHT:    -1.8 rad/s
STOP:      0
```

Forward and backward are relative to the car's current yaw, not fixed world X. That means if the car rotates, forward still means "move in the direction the car is facing."

The relevant runtime dependencies were added to `package.xml`:

```xml
<exec_depend>rclpy</exec_depend>
<exec_depend>gazebo_ros</exec_depend>
<exec_depend>gazebo_msgs</exec_depend>
<exec_depend>geometry_msgs</exec_depend>
```

The script is installed through `CMakeLists.txt`:

```cmake
install(
  PROGRAMS scripts/gesture_gazebo_driver
  DESTINATION lib/${PROJECT_NAME}
)
```

## Build Commands

From WSL Ubuntu-20.04 as `ros_joe`:

```bash
cd ~/HandGestureDrone_ws
source /opt/ros/foxy/setup.bash
colcon build --packages-select car_assembly_2nd
source install/setup.bash
```

Verify that ROS can find the executable:

```bash
ros2 pkg executables car_assembly_2nd
```

Expected:

```text
car_assembly_2nd gesture_gazebo_driver
```

## Final Run Commands

The ROS/Gazebo side can now be started with one ROS 2 launch file.

### One-Command ROS/Gazebo Launch

From WSL Ubuntu-20.04 as `ros_joe`:

```bash
source /opt/ros/foxy/setup.bash
source ~/HandGestureDrone_ws/install/setup.bash
ros2 launch car_assembly_2nd gesture_car.launch.py
```

This launch file starts:

```text
Gazebo with gesture_empty.world
spawn_entity.py for car_assembly_2nd
gesture_gazebo_driver on UDP port 4210
```

The Windows webcam recognizer still runs separately in PowerShell because it uses the Windows venv and camera.

You can override launch settings like this:

```bash
ros2 launch car_assembly_2nd gesture_car.launch.py turn_speed:=2.5 udp_port:=4210
```

### Manual Terminal Flow

The old manual flow is still useful for debugging. Use separate terminals.

### Terminal 1: Start Gazebo

```bash
source /opt/ros/foxy/setup.bash
source ~/HandGestureDrone_ws/install/setup.bash

export GAZEBO_MODEL_DATABASE_URI=""
export LIBGL_ALWAYS_SOFTWARE=1
export QT_X11_NO_MITSHM=1

gazebo --verbose \
  ~/HandGestureDrone_ws/install/car_assembly_2nd/share/car_assembly_2nd/worlds/gesture_empty.world \
  -s libgazebo_ros_init.so \
  -s libgazebo_ros_factory.so
```

### Terminal 2: Spawn The Car

```bash
source /opt/ros/foxy/setup.bash
source ~/HandGestureDrone_ws/install/setup.bash

ros2 run gazebo_ros spawn_entity.py \
  -entity car_assembly_2nd \
  -file ~/HandGestureDrone_ws/src/car_assembly_2nd/urdf/car_assembly_2nd.urdf
```

### Terminal 3: Start The Gesture Driver

```bash
source /opt/ros/foxy/setup.bash
source ~/HandGestureDrone_ws/install/setup.bash

ros2 run car_assembly_2nd gesture_gazebo_driver
```

Expected driver logs:

```text
Listening for UDP commands on 0.0.0.0:4210
Controlling Gazebo model 'car_assembly_2nd'
Gazebo entity state services are ready
Active command: STOP source=timeout
```

### Terminal 4: Start The Windows Recognizer

In PowerShell:

```powershell
cd "C:\Users\Yoav\Desktop\Personal Projects\hand-gestured-car"
.\.venv\Scripts\activate
Remove-Item Env:GESTURE_BRIDGE_HOST -ErrorAction SilentlyContinue
python .\src\live_recognize.py
```

Expected recognizer log:

```text
[Live Recognize] Running.
Controls: Q quit | A toggle armed | circles arm | STOP disarms

[Gesture Bridge] Sending UDP commands to 172.22.129.75:4210 (auto WSL Ubuntu-20.04/ros_joe eth0)
```

The IP can change after restarting WSL, but the script now auto-detects it.

## The Routing Problem We Found

The confusing part was that the recognizer appeared to work:

```text
[ARMED] via CIRCLE_CW
[EVENT] PUSH -> FORWARD
[EVENT] PULL -> BACKWARD
```

But the car did not move.

The first bad sign was:

```text
[Gesture Bridge] Sending UDP commands to 127.0.0.1:4210
```

On Windows, `127.0.0.1` means Windows itself. It does not necessarily reach the UDP socket inside WSL.

Then we changed the recognizer to auto-detect WSL, but it initially detected the wrong WSL instance:

```text
172.22.128.1
```

The real ROS/Gazebo system was running in:

```text
Ubuntu-20.04
user: ros_joe
```

But plain `wsl` was entering a different default distro/user:

```text
Ubuntu-18.04
home: /home/yoavmatalon
```

So the script was detecting an IP from the wrong Linux environment.

The correct check was:

```powershell
wsl -d Ubuntu-20.04 -u ros_joe hostname -I
```

That returned:

```text
172.22.129.75
```

The recognizer was patched so `resolve_bridge_host()` explicitly checks:

```text
distro: Ubuntu-20.04
user:   ros_joe
```

It now prints:

```text
[Gesture Bridge] Sending UDP commands to 172.22.129.75:4210 (auto WSL Ubuntu-20.04/ros_joe eth0)
```

## Manual UDP Test That Proved It Worked

We manually sent UDP packets from Windows to the corrected WSL IP:

```powershell
@'
import json
import socket
import time

addr = ("172.22.129.75", 4210)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

for cmd, duration in [("FORWARD", 1.5), ("STOP", 0.3)]:
    msg = json.dumps({
        "cmd": cmd,
        "gesture": "manual",
        "confidence": 1.0,
        "hand": "test",
        "ts": time.time()
    }).encode("utf-8")

    end = time.time() + duration
    while time.time() < end:
        sock.sendto(msg, addr)
        time.sleep(0.05)

print("sent manual FORWARD then STOP to", addr)
'@ | python -
```

Before the test, Gazebo reported the model near:

```text
x=-0.1428, y=-0.0804
```

After the test, Gazebo reported:

```text
x=0.0450, y=-0.0330
```

That proved:

- Gazebo had the model.
- `/get_entity_state` worked.
- `/set_entity_state` worked.
- The driver was receiving UDP.
- The car moved when packets were sent to the correct IP.

## Useful Debug Commands

Check the correct WSL IP:

```powershell
wsl -d Ubuntu-20.04 -u ros_joe hostname -I
```

Check which WSL distro is default:

```powershell
wsl -l -v
```

Check whether Gazebo and the driver are running:

```powershell
wsl -d Ubuntu-20.04 -u ros_joe sh -lc "ps -ef | grep -E 'gesture_gazebo_driver|gazebo|gzserver|gzclient|spawn_entity' | grep -v grep"
```

Check whether the UDP port is already in use in the correct WSL distro:

```powershell
wsl -d Ubuntu-20.04 -u ros_joe python3 - <<'PY'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.bind(("0.0.0.0", 4210))
    print("udp_4210_free")
except OSError as e:
    print("udp_4210_in_use", e)
PY
```

If the driver is running, expected output is:

```text
udp_4210_in_use [Errno 98] Address already in use
```

Check Gazebo state services:

```bash
source /opt/ros/foxy/setup.bash
source ~/HandGestureDrone_ws/install/setup.bash
ros2 service list | grep -E '/get_entity_state|/set_entity_state'
```

Expected:

```text
/get_entity_state
/set_entity_state
```

Check whether Gazebo can see the car model:

```bash
source /opt/ros/foxy/setup.bash
source ~/HandGestureDrone_ws/install/setup.bash
ros2 service call /get_entity_state gazebo_msgs/srv/GetEntityState "{name: car_assembly_2nd, reference_frame: world}"
```

Expected:

```text
success=True
```

## Manual Override For The Bridge IP

If auto-detection fails, set the host manually in PowerShell:

```powershell
$env:GESTURE_BRIDGE_HOST="172.22.129.75"
python .\src\live_recognize.py
```

To go back to auto-detect:

```powershell
Remove-Item Env:GESTURE_BRIDGE_HOST -ErrorAction SilentlyContinue
python .\src\live_recognize.py
```

The script also supports these optional environment variables:

```powershell
$env:GESTURE_WSL_DISTRO="Ubuntu-20.04"
$env:GESTURE_WSL_USER="ros_joe"
$env:GESTURE_BRIDGE_PORT="4210"
```

## Current Limitations

This is a V1 control loop. It prioritizes "gesture makes the Gazebo model move" over physical realism.

Current limitations:

- The car is moved with direct Gazebo entity velocity, not wheel physics.
- Wheels do not yet drive the car through contact/friction.
- There is no differential drive plugin yet.
- There is no ROS 2 control hardware interface yet.
- If the hand is held still, `STOP_IDLE` can disarm the bridge.
- The recognizer process must be restarted after changing environment variables or code.

## Next Improvements

Good next steps:

- Add a small on-screen status indicator for the selected UDP host and armed state.
- Add a console print when each UDP heartbeat is sent, only while debugging.
- Add a ROS topic output like `/gesture_cmd` for visibility in `ros2 topic echo`.
- Convert direct Gazebo control to a differential drive setup.
- Connect wheel joints to a Gazebo plugin or ROS 2 control.
- Add speed parameters to the driver launch file.
- Add a proper launch file that starts Gazebo, spawns the car, and starts the driver together.

## Short Version

The project works because these pieces are now aligned:

1. The URDF package is a valid ROS 2 package.
2. Mesh paths resolve through `package://car_assembly_2nd/...`.
3. Gazebo can find installed model assets through the environment hook.
4. The custom Gazebo world loads `libgazebo_ros_state.so`.
5. The ROS driver listens on UDP port `4210`.
6. The Windows recognizer sends commands to the correct WSL Ubuntu-20.04 IP.
7. The driver converts gesture commands into Gazebo model velocity.

The final key fix was not the gesture model. It was routing: the recognizer was sending commands to the wrong address because Windows had multiple WSL environments. Once it explicitly targeted `Ubuntu-20.04` as `ros_joe`, the UDP packets reached the running ROS driver and the car moved.
