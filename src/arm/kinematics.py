import numpy as np
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink

# Constants for YF-6125MG Servos
# Joints 1-5: physical 0° to 270°, neutral at 135°, IK bounds = ±135°
# Gripper (Joint 6): physical 60° to 180°, neutral at 120°. Handled as a separate action, 
# but we define its physical length as a fixed offset (TCP).
JOINT_OFFSET_DEG = [0, 135, 135, 135, 135, 135, 120]  # index 0 is OriginLink

class ArmKinematics:
    """
    Inverse Kinematics engine for the custom 5-DOF YF-6125MG manipulator.
    Coordinates are in meters. Zero-position is straight up along Z-axis.
    """
    def __init__(self):
        self.chain = Chain(name="yf_6125mg_arm", links=[
            OriginLink(),

            URDFLink(
                name="base_rotation",           # CH1: 0°-270° (Yaw)
                origin_translation=[0, 0, 0.042],
                origin_orientation=[0, 0, 0],
                rotation=[0, 0, 1],
                bounds=(-np.radians(135), np.radians(135)),
            ),

            URDFLink(
                name="shoulder",                # CH2: 0°-270° (Pitch)
                origin_translation=[0, 0, 0.105],
                origin_orientation=[0, 0, 0],
                rotation=[0, 1, 0],
                bounds=(-np.radians(135), np.radians(135)),
            ),

            URDFLink(
                name="elbow",                   # CH3: 0°-270° (Pitch)
                origin_translation=[0, 0, 0.1275],
                origin_orientation=[0, 0, 0],
                rotation=[0, 1, 0],
                bounds=(-np.radians(135), np.radians(135)),
            ),

            URDFLink(
                name="wrist_pitch",             # CH4: 0°-270° (Pitch)
                origin_translation=[0, 0, 0.070],
                origin_orientation=[0, 0, 0],
                rotation=[0, 1, 0],
                bounds=(-np.radians(135), np.radians(135)),
            ),

            URDFLink(
                name="wrist_rotate",            # CH5: 0°-270° (Roll)
                origin_translation=[0, 0, 0.031],
                origin_orientation=[0, 0, 0],
                rotation=[0, 0, 1],
                bounds=(-np.radians(135), np.radians(135)),
            ),

            # Tool Center Point (TCP). It represents the tip of the gripper.
            URDFLink(
                name="gripper_tcp",             # CH6 Handled elsewhere (Claw state)
                origin_translation=[0, 0, 0.083],
                origin_orientation=[0, 0, 0],
                rotation=None,             # Rotation [0,0,0] makes it a fixed link in ikpy
                joint_type="fixed",             # Declare this correctly for ikpy solver
            ),
        ])

    def calculate_servo_angles(self, target_xyz: list, target_orientation=None) -> list:
        """
        Calculate Inverse Kinematics and map the ±135° IK output back to 
        the physical 0°-270° target positions for the servos.

        Parameters:
            target_xyz: [x, y, z] target location in meters.
            target_orientation: Optional target orientation vector.
            
        Returns:
            List of 5 servo angles [CH1, CH2, CH3, CH4, CH5] in degrees.
        """
        # Run Inverse Kinematics solver
        if target_orientation is not None:
            ik_angles_rad = self.chain.inverse_kinematics(
                target_position=target_xyz, 
                target_orientation=target_orientation, 
                orientation_mode="all"
            )
        else:
            ik_angles_rad = self.chain.inverse_kinematics(target_position=target_xyz)

        servo_angles_deg = []
        
        # Convert [-pi, pi] bounded outputs to [0, 270] physical degrees
        # Note: ik_angles_rad includes the OriginLink at index 0 and Gripper at index 6
        for i in range(1, 6):  # Only map CH1 to CH5
            angle_deg = np.degrees(ik_angles_rad[i]) + JOINT_OFFSET_DEG[i]
            
            # Clip safely to hardware limits (0 - 270 degrees)
            angle_deg = max(0.0, min(270.0, angle_deg))
            servo_angles_deg.append(round(angle_deg, 2))

        return servo_angles_deg
