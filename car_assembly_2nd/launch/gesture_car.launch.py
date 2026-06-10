import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_name = "car_assembly_2nd"
    package_share = get_package_share_directory(package_name)

    default_world = os.path.join(package_share, "worlds", "gesture_empty.world")
    default_urdf = os.path.join(package_share, "urdf", "car_assembly_2nd.urdf")

    model_name = LaunchConfiguration("model_name")
    urdf_file = LaunchConfiguration("urdf_file")
    world_file = LaunchConfiguration("world_file")
    udp_port = LaunchConfiguration("udp_port")
    turn_speed = LaunchConfiguration("turn_speed")

    gazebo = ExecuteProcess(
        cmd=[
            "gazebo",
            "--verbose",
            world_file,
            "-s",
            "libgazebo_ros_init.so",
            "-s",
            "libgazebo_ros_factory.so",
        ],
        output="screen",
    )

    spawn_car = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        name="spawn_car_assembly_2nd",
        output="screen",
        arguments=[
            "-entity",
            model_name,
            "-file",
            urdf_file,
        ],
    )

    gesture_driver = Node(
        package=package_name,
        executable="gesture_gazebo_driver",
        name="gesture_gazebo_driver",
        output="screen",
        parameters=[
            {
                "model_name": model_name,
                "udp_port": ParameterValue(udp_port, value_type=int),
                "turn_speed": ParameterValue(turn_speed, value_type=float),
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("model_name", default_value=package_name),
            DeclareLaunchArgument("world_file", default_value=default_world),
            DeclareLaunchArgument("urdf_file", default_value=default_urdf),
            DeclareLaunchArgument("udp_port", default_value="4210"),
            DeclareLaunchArgument("turn_speed", default_value="1.8"),
            SetEnvironmentVariable("GAZEBO_MODEL_DATABASE_URI", ""),
            SetEnvironmentVariable("LIBGL_ALWAYS_SOFTWARE", "1"),
            SetEnvironmentVariable("QT_X11_NO_MITSHM", "1"),
            gazebo,
            TimerAction(period=3.0, actions=[spawn_car]),
            TimerAction(period=6.0, actions=[gesture_driver]),
        ]
    )
