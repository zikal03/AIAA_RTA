# rta
RTA ROS package for runtime assurance simulation using ROS 2, Gazebo, PX4, and QGroundControl.

## Dependencies

- ROS 2 (Humble or later)
- `rclcpp`, `ros_gz_sim`, `ros_gz_bridge`, `ros_gz_interfaces`
- `px4_msgs`
- `mavros`

## Installation

### 1. Clone the repository

Navigate to your ROS 2 workspace `src` directory and clone the package:

```bash
cd ~/ros2_ws/src
git clone <repository-url> rta
```

### 2. Build the package

From the workspace root, build using colcon:

```bash
cd ~/ros2_ws
colcon build --packages-select rta --symlink-install
```

### 3. Source the workspace

```bash
source install/setup.bash
```

## Running the Package

Open three separate terminals. Source the workspace in each terminal before running:

```bash
source ~/ros2_ws/install/setup.bash
```

### Terminal 1 — Launch the RTA scenario

```bash
ros2 launch rta rta_scenarioA.launch.py
```

### Terminal 2 — Launch QGroundControl

```bash
~/QGroundControl.AppImage
```

### Terminal 3 — Run the RTA node

```bash
ros2 run rta Roll_RTA.py
```

