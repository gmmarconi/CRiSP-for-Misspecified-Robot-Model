# Helper function returning a customized URDF of the 5R planar manipulator
import numpy as np


def generate_custom_urdf(joint_bias: np.ndarray = np.array((0, 0, 0, 0, 0)),
                         link_bias: np.ndarray = np.array((0, 0, 0, 0, 0))) -> str:
    robot_urdf = f"""
        <?xml version="1.0" ?>
        <robot name = "5d_planar_manipulator_v2">
            <!-- links section -->>
            <link name = "base_link">
                <inertial>
                    <origin xyz = "0 0 0" />
                    <mass value = "0.5" />
                    <inertia ixx = "0.5" iyy = "0.5" izz = "0.5" ixy = "0" ixz = "0" iyz = "0" />
                </inertial>
                <visual>
                    <origin xyz = "0 0 0"  rpy="1.5708 0 0"/>
                    <geometry>
                <cylinder radius = "0.2" length = "0.2"  />
                    </geometry>
                    <material name = "dark_gray">
                        <color rgba = "0.1 0.1 0.1 1" />
                    </material>
                </visual>
            </link>
            <link name = "link_1">
                <inertial>
                    <origin xyz = "0 0 1" />
                    <mass value = "0.5" />
                    <inertia ixx = "0.5" iyy = "0.5" izz = "0.5" ixy = "0" ixz = "0" iyz = "0" />
                </inertial>
                <visual>
                    <origin xyz = "{np.sin(joint_bias[0])} 0 {np.cos(joint_bias[0])}"  rpy="0 {joint_bias[0]} 0" />
                    <geometry>
                        <cylinder radius = "0.1" length = "{2+link_bias[0]}"  />
                    </geometry>
                    <material name = "light_gray">
                        <color rgba = "0.3 0.3 0.3 1" />
                    </material>
                </visual>
            </link>
            <link name = "link_2">
                <inertial>
                    <origin xyz = "{np.sin(joint_bias[1])} 0 {np.cos(joint_bias[1])}" rpy="0 {joint_bias[1]} 0" />
                    <mass value = "0.5" />
                    <inertia ixx = "0.5" iyy = "0.5" izz = "0.5" ixy = "0" ixz = "0" iyz = "0" />
                </inertial>
                <visual>
                    <origin xyz = "{np.sin(joint_bias[1])} 0 {np.cos(joint_bias[1])}"  rpy="0 {joint_bias[1]} 0" />
                    <geometry>
                        <cylinder radius = "0.1" length = "{2+link_bias[1]}"  />
                    </geometry>
                        <material name = "light_gray">
                        <color rgba = "0.3 0.3 0.3 1" />
                    </material>
                </visual>
            </link>
            <link name = "link_3">
                <inertial>
                  <origin xyz = "0 0 1" />
                  <mass value = "0.5" />
                  <inertia ixx = "0.5" iyy = "0.5" izz = "0.5" ixy = "0" ixz = "0" iyz = "0" />
                </inertial>
                <visual>
                  <origin xyz = "{np.sin(joint_bias[2])} 0 {np.cos(joint_bias[2])}"  rpy="0 {joint_bias[2]} 0" />
                  <geometry>
                    <cylinder radius = "0.1" length = "{2+link_bias[2]}"  />
                  </geometry>
                  <material name = "light_gray">
                    <color rgba = "0.3 0.3 0.3 1" />
                  </material>
                </visual>
            </link>
            <link name = "link_4">
                <inertial>
                  <origin xyz = "0 0 1" />
                  <mass value = "0.5" />
                  <inertia ixx = "0.5" iyy = "0.5" izz = "0.5" ixy = "0" ixz = "0" iyz = "0" />
                </inertial>
                <visual>
                  <origin xyz = "{np.sin(joint_bias[3])} 0 {np.cos(joint_bias[3])}"  rpy="0 {joint_bias[3]} 0" />
                  <geometry>
                    <cylinder radius = "0.1" length = "{2+link_bias[3]}"  />
                  </geometry>
                  <material name = "light_gray">
                    <color rgba = "0.3 0.3 0.3 1" />
                  </material>
                </visual>
            </link>
            <link name = "link_5">
                <inertial>
                  <origin xyz = "0 0 1" />
                  <mass value = "0.5" />
                  <inertia ixx = "0.5" iyy = "0.5" izz = "0.5" ixy = "0" ixz = "0" iyz = "0" />
                </inertial>
                <visual>
                  <origin xyz = "{np.sin(joint_bias[4])} 0 {np.cos(joint_bias[4])}"  rpy="0 {joint_bias[4]} 0" />
                  <geometry>
                    <cylinder radius = "0.1" length = "{2+link_bias[4]}"  />
                  </geometry>
                  <material name = "light_gray">
                    <color rgba = "0.3 0.3 0.3 1" />
                  </material>
                </visual>
            </link>
              <link name = "link_6">
                <inertial>
                  <origin xyz = "0 0 1" />
                  <mass value = "0.5" />
                  <inertia ixx = "0.5" iyy = "0.5" izz = "0.5"
              ixy = "0" ixz = "0" iyz = "0" />
                </inertial>
                <visual>
                  <origin xyz = "0 0 1" />
                  <geometry>
                    <cylinder radius = "0.1" length = "2"  />
                  </geometry>
                  <material name = "light_gray">
                    <color rgba = "0.3 0.3 0.3 1" />
                  </material>
                </visual>
            </link>
        
            <!-- joints section -->>
            <joint name = "joint_b_1" type = "continuous">
                <parent link = "base_link" />
                <child link = "link_1" />
                <origin xyz = "0 0 0" />
                <axis xyz = "0 -1 0" />
            <limit lower = "0" upper = "3.1415" />
            </joint>
            <joint name = "joint_1_2" type = "continuous">
                <parent link = "link_1" />
                <child link = "link_2" />
                <origin xyz = "{(2+link_bias[0])*np.sin(joint_bias[0])} 0 {(2+link_bias[0])*np.cos(joint_bias[0])}"  rpy="0 {joint_bias[0]} 0" />
                <axis xyz = "0 -1 0" />
            <limit lower = "-3.1415" upper = "0" />
            </joint>
            <joint name = "joint_2_3" type = "continuous">
                <parent link = "link_2" />
                <child link = "link_3" />
                <origin xyz = "{(2+link_bias[1])*np.sin(joint_bias[1])} 0 {(2+link_bias[1])*np.cos(joint_bias[1])}"  rpy="0 {joint_bias[1]} 0" />
                <axis xyz = "0 -1 0" />
            <limit lower = "-1.5707" upper = " 1.5707" />
            </joint>
            <joint name = "joint_3_4" type = "continuous">
                <parent link = "link_3" />
                <child link = "link_4" />
                <origin xyz = "{(2+link_bias[2])*np.sin(joint_bias[2])} 0 {(2+link_bias[2])*np.cos(joint_bias[2])}"  rpy="0 {joint_bias[2]} 0" />
                <axis xyz = "0 -1 0" />
            <limit lower = "-3.1415" upper = "0" />
            </joint>
            <joint name = "joint_4_5" type = "continuous">
                <parent link = "link_4" />
                <child link = "link_5" />
                <origin xyz = "{(2+link_bias[3])*np.sin(joint_bias[3])} 0 {(2+link_bias[3])*np.cos(joint_bias[3])}"  rpy="0 {joint_bias[3]} 0" />
                <axis xyz = "0 -1 0" />
            <limit lower = "-1.5707" upper = "1.5707" />
            </joint>
            <joint name = "joint_5_6" type = "fixed">
                <parent link = "link_5" />
                <child link = "link_6" />
                <origin xyz = "{(2+link_bias[4])*np.sin(joint_bias[4])} 0 {(2+link_bias[4])*np.cos(joint_bias[4])}"  rpy="0 {joint_bias[4]} 0" />
                <axis xyz = "0 -1 0" />
            </joint>
        </robot>"""
    return robot_urdf

