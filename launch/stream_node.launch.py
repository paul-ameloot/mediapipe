import launch
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Webcam Args
        launch.actions.DeclareLaunchArgument('webcam', default_value='0'),
        launch.actions.DeclareLaunchArgument('realsense', default_value='false'),

        # Enable Mediapipe Modules Args
        launch.actions.DeclareLaunchArgument('enable_right_hand', default_value='true'),
        launch.actions.DeclareLaunchArgument('enable_left_hand', default_value='true'),
        launch.actions.DeclareLaunchArgument('enable_pose', default_value='true'),
        launch.actions.DeclareLaunchArgument('enable_face', default_value='false'),
        launch.actions.DeclareLaunchArgument('enable_face_detection', default_value='false'),
        launch.actions.DeclareLaunchArgument('enable_objectron', default_value='false'),

        # FaceMesh Connections Mode -> 0: Contours, 1: Tesselation
        launch.actions.DeclareLaunchArgument('face_mesh_mode', default_value='0'),

        # Available Models = 'Shoe', 'Chair', 'Cup', 'Camera'
        launch.actions.DeclareLaunchArgument('objectron_model', default_value='Shoe'),

        # Webcam Parameters
        launch_ros.actions.SetLaunchConfiguration('mediapipe_gesture_recognition/webcam', launch.substitutions.LaunchConfiguration('webcam')),
        launch_ros.actions.SetLaunchConfiguration('mediapipe_gesture_recognition/realsense', launch.substitutions.LaunchConfiguration('realsense')),

        # Enable Mediapipe Modules Parameters
        launch_ros.actions.SetLaunchConfiguration('mediapipe_gesture_recognition/enable_right_hand', launch.substitutions.LaunchConfiguration('enable_right_hand')),
        launch_ros.actions.SetLaunchConfiguration('mediapipe_gesture_recognition/enable_left_hand', launch.substitutions.LaunchConfiguration('enable_left_hand')),
        launch_ros.actions.SetLaunchConfiguration('mediapipe_gesture_recognition/enable_pose', launch.substitutions.LaunchConfiguration('enable_pose')),
        launch_ros.actions.SetLaunchConfiguration('mediapipe_gesture_recognition/enable_face', launch.substitutions.LaunchConfiguration('enable_face')),
        launch_ros.actions.SetLaunchConfiguration('mediapipe_gesture_recognition/enable_face_detection', launch.substitutions.LaunchConfiguration('enable_face_detection')),
        launch_ros.actions.SetLaunchConfiguration('mediapipe_gesture_recognition/enable_objectron', launch.substitutions.LaunchConfiguration('enable_objectron')),
        launch_ros.actions.SetLaunchConfiguration('mediapipe_gesture_recognition/objectron_model', launch.substitutions.LaunchConfiguration('objectron_model')),
        launch_ros.actions.SetLaunchConfiguration('mediapipe_gesture_recognition/face_mesh_mode', launch.substitutions.LaunchConfiguration('face_mesh_mode')),

        # Load Stream Node YAML Config File
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                'path/to/mediapipe_gesture_recognition/config/stream_node.py'
            )
        ),

        # Mediapipe Stream Node
        launch_ros.actions.Node(
            package='mediapipe_gesture_recognition',
            executable='stream_node.py',
            name='mediapipe_stream_node',
            output='screen'
        ),
    ])