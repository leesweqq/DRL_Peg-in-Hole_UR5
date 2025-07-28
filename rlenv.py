import pybullet as p
import pybullet_data
import numpy as np
import time
import random
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Dict, Box
from collections import namedtuple
import matplotlib.pyplot as plt
import math

class PegInHoleGymEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.physics_client = p.connect(p.GUI)  # Connect to PyBullet in GUI mode
        
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        self.image_width =  100
        self.image_height = 100

        self.max_steps = 500
        self.step_count = 0

        self._load_env()
        self._load_robot()
        self._parse_joint_info()

        # Define observation space with a grayscale camera image
        self.observation_space = Dict({
            "cam_image": Box(
                low=0,
                high=255,
                shape=(self.image_height, self.image_width, 1),
                dtype=np.uint8
            )
        })

        # Define action space: 3D continuous small movements
        self.action_space = spaces.Box(low=-0.005, high=0.005, shape=(3,), dtype=np.float32)

        self.reset()
        
        p.resetDebugVisualizerCamera(
            cameraDistance=1.0,    
            cameraYaw=110,            
            cameraPitch=-45,         
            cameraTargetPosition=[0.5, 0, 0.5]  
        )

    def _load_env(self):
        # Load the plane and the table into the simulation
        p.loadURDF("plane.urdf")
        self.table_id = p.loadURDF("table/table.urdf", [0.4, 0, 0], p.getQuaternionFromEuler([0, 0, np.pi / 2]))

    def _load_robot(self):
        # Load the UR5 robot with a Robotiq gripper
        self.robot_id = p.loadURDF("./urdf/ur5_robotiq_85.urdf", [0, 0, 0.62],
                                   p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)
        self.eef_link_index = 6  # End-effector link index
        num_joints = p.getNumJoints(self.robot_id)
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i)
            print(f"Index: {info[0]}, Name: {info[1].decode('utf-8')}")

    def _parse_joint_info(self):
        # Extract joint information and identify controllable joints
        jointInfo = namedtuple('jointInfo', ['id', 'name', 'type', 'lowerLimit', 'upperLimit',
                                             'maxForce', 'maxVelocity', 'controllable'])
        self.joints = []
        self.controllable_joints = []
        for i in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[2]
            lowerLimit = info[8]
            upperLimit = info[9]
            maxForce = info[10]
            maxVelocity = info[11]
            controllable = jointType != p.JOINT_FIXED
            self.joints.append(jointInfo(jointID, jointName, jointType, lowerLimit,
                                         upperLimit, maxForce, maxVelocity, controllable))
            if controllable:
                self.controllable_joints.append(jointID)
        self.arm_joints = self.controllable_joints[:6]
        self.rest_poses = [0, -1.57, 1.57, -1.5, -1.57, 0.0]

    def update_simulation(self, steps, sleep_time=0.01):
        # Step simulation multiple times
        for _ in range(steps):
            p.stepSimulation()
    
    def reset(self, seed=None, options=None):
        self.step_count = 0

        # Reset arm to default resting poses
        for i, joint_id in enumerate(self.arm_joints):
            p.setJointMotorControl2(self.robot_id, joint_id, p.POSITION_CONTROL, self.rest_poses[i], force=500)
        for _ in range(100):
            p.stepSimulation()

        # Remove existing hole object if exists
        if hasattr(self, 'hole_id'):
            p.removeBody(self.hole_id)

        # Randomize hole position
        x = random.uniform(0.5, 0.6)
        y = random.uniform(0, 0.1)
        z = 0.65
        self.target_pos = [x, y, z]
        self.hole_id = p.loadURDF("./urdf/box.urdf", self.target_pos, p.getQuaternionFromEuler([0, 0, 0]))

        return self._get_obs(), {}

    def move_arm_to(self, pos, orn):
        # Move the robot arm to the target pose using inverse kinematics
        joint_poses = p.calculateInverseKinematics(
            self.robot_id,
            self.eef_link_index,
            pos,
            orn,
            restPoses=[p.getJointState(self.robot_id, j)[0] for j in self.arm_joints],
            lowerLimits=[self.joints[j].lowerLimit for j in self.arm_joints],
            upperLimits=[self.joints[j].upperLimit for j in self.arm_joints],
            jointRanges=[self.joints[j].upperLimit - self.joints[j].lowerLimit for j in self.arm_joints],
        )
        for i, joint_id in enumerate(self.arm_joints):
            p.setJointMotorControl2(self.robot_id, joint_id, p.POSITION_CONTROL, joint_poses[i], force=500)

    def render_observation(self, observation):
        cam_image = observation["cam_image"]
        plt.imshow(cam_image.squeeze(), cmap="gray")
        plt.title(f"Step {self.step_count} - Observation Image")
        plt.axis("off")
        plt.pause(0.01)  
        plt.clf()  

    def step(self, action):
        self.step_count += 1
        info ={"insertion_success": False}  
        # Clip the action within valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        current = p.getLinkState(self.robot_id, self.eef_link_index)
        current_pos = current[0]
        current_orn = current[1]

        # Calculate new position and move the robot
        new_pos = [current_pos[i] + action[i] for i in range(3)]
        self.move_arm_to(new_pos, current_orn)
        self.update_simulation(10)

        obs = self._get_obs()
        # #plot observation image
        # self.render_observation(obs)
        reward, dist_xy, dist_z = self._compute_reward()
        collided = self._check_collision()


        if collided:
            print("⬅️Collision detected - resetting")
            done = True
        else:
            done = self._check_done()

        if self._check_inserted():
            print("✅ Insertion successful")
            info={"insertion_success": True}

        print(f"Step {self.step_count} | XY distance: {dist_xy:.5f} | Z distance: {dist_z:.5f} | Reward: {reward:.2f}")

        truncated = self.step_count >= self.max_steps

        return obs, reward, done, truncated, info

    def _check_collision(self):
        # Check if robot collides with table or hole, unless close enough to insert
        eef_pos = p.getLinkState(self.robot_id, self.eef_link_index)[0]
        dist_xy = np.linalg.norm(np.array(eef_pos[:2]) - np.array(self.target_pos[:2]))
        close_enough = dist_xy < 0.02 and (abs(eef_pos[2] - self.target_pos[2]) < 0.15)

        if close_enough:
            return False

        contact_with_table = len(p.getContactPoints(self.robot_id, self.table_id)) > 0
        contact_with_hole = len(p.getContactPoints(self.robot_id, self.hole_id)) > 0

        return contact_with_table or contact_with_hole

    def _get_obs(self):
        # Get grayscale image observation from camera attached to robot
        self.camera_link_index = -1
        for i in range(p.getNumJoints(self.robot_id)):
            link_name = p.getJointInfo(self.robot_id, i)[12].decode('utf-8')
            if link_name == "camera_link":
                self.camera_link_index = i
                break
        rgba = self._render_camera_from_link(self.robot_id, self.camera_link_index)
        gray = np.mean(rgba[:, :, :3], axis=2).astype(np.uint8)
        gray = np.expand_dims(gray, axis=2)
        return {"cam_image": gray}

    def _render_camera_from_link(self, body_id, link_index):
        # Render image from specified robot link (e.g., camera)
        link_state = p.getLinkState(body_id, link_index, computeForwardKinematics=True)
        link_pos = link_state[0]
        link_ori = link_state[1]

        rot_matrix = p.getMatrixFromQuaternion(link_ori)
        forward = [-rot_matrix[2], -rot_matrix[5], -rot_matrix[8]]
        up = [rot_matrix[0], rot_matrix[3], rot_matrix[6]]

        cam_eye = link_pos
        cam_target = [cam_eye[0] + forward[0] * 0.2,
                      cam_eye[1] + forward[1] * 0.2,
                      cam_eye[2] + forward[2] * 0.2]

        view = p.computeViewMatrix(cam_eye, cam_target, up)
        proj = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=self.image_width / self.image_height,
            nearVal=0.01,
            farVal=3.0
        )

        w, h, rgba, _, _ = p.getCameraImage(
            self.image_width,
            self.image_height,
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        rgba_img = np.reshape(rgba, (h, w, 4))
        return rgba_img

    def _compute_reward(self):
        # Compute reward based on distance to target
        eef_pos = p.getLinkState(self.robot_id, self.eef_link_index)[0]
        dist_xy = np.linalg.norm(np.array(eef_pos[:2]) - np.array(self.target_pos[:2]))
        dist_z = abs(eef_pos[2] - self.target_pos[2])
        
        reward = 10 * (-dist_xy - 0.5 * dist_z)

        if self._check_inserted():
            reward += 100.0  # High reward for successful insertion

        return reward, dist_xy, dist_z

    def _check_done(self):
        # Check if task is done
        return self._check_inserted()

    def _check_inserted(self):
        # Check if peg is successfully inserted into the hole
        eef_pos = p.getLinkState(self.robot_id, self.eef_link_index)[0]
        dist_xy = np.linalg.norm(np.array(eef_pos[:2]) - np.array(self.target_pos[:2]))
        close_enough = dist_xy < 0.01 and (abs(eef_pos[2] - self.target_pos[2]) < 0.1)
        return close_enough

    def render(self):
        pass

    def close(self):
        p.disconnect()
