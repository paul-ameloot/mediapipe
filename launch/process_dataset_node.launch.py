import launch
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        # Enable Mediapipe Modules Args
        DeclareLaunchArgument('enable_right_hand', default_value='true'),
        DeclareLaunchArgument('enable_left_hand', default_value='true'),
        DeclareLaunchArgument('enable_pose', default_value='true'),
        DeclareLaunchArgument('enable_face', default_value='false'),
        DeclareLaunchArgument('debug', default_value='true'),

        # Enable Mediapipe Modules Parameters
        Node(
            package='mediapipe_gesture_recognition',
            executable='process_dataset_node.py',
            name='process_dataset_node',
            output='screen',
            parameters=[
                {'enable_right_hand': LaunchConfiguration('enable_right_hand')},
                {'enable_left_hand': LaunchConfiguration('enable_left_hand')},
                {'enable_pose': LaunchConfiguration('enable_pose')},
                {'enable_face': LaunchConfiguration('enable_face')},
                {'debug': LaunchConfiguration('debug')}
            ]
        ),
    ])