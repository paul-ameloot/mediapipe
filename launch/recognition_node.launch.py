import launch
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Webcam Args
        launch.actions.DeclareLaunchArgument('realsense', default_value='false'),

        # Recognition Precision Arg / Param
        launch.actions.DeclareLaunchArgument('recognition_precision_probability', default_value='0.8'),
        launch_ros.actions.Node(
            package='mediapipe_gesture_recognition',
            executable='recognition_node.py',
            name='mediapipe_3D_recognition_node',
            output='screen',
            parameters=[{'recognition_precision_probability': launch.substitutions.LaunchConfiguration('recognition_precision_probability')}]
        ),

        # Point-At Area - Raw Function Node
        launch_ros.actions.Node(
            condition=launch.conditions.IfCondition(launch.substitutions.LaunchConfiguration('realsense')),
            package='mediapipe_gesture_recognition',
            executable='3Dpoint_node_area.py',
            name='point_area_node',
            output='screen'
        ),
    ])