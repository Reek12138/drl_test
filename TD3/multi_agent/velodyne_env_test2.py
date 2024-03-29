import math
import os
import random
import subprocess
import time
from os import path

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

GOAL_REACHED_DIST = 0.3
COLLISION_DIST = 0.35
TIME_DELTA = 0.1


# Check if the random goal position is located on an obstacle and do not accept it if it is
def check_pos(x, y):
    goal_ok = True

    if -3.8 > x > -6.2 and 6.2 > y > 3.8:
        goal_ok = False

    if -1.3 > x > -2.7 and 4.7 > y > -0.2:
        goal_ok = False

    if -0.3 > x > -4.2 and 2.7 > y > 1.3:
        goal_ok = False

    if -0.8 > x > -4.2 and -2.3 > y > -4.2:
        goal_ok = False

    if -1.3 > x > -3.7 and -0.8 > y > -2.7:
        goal_ok = False

    if 4.2 > x > 0.8 and -1.8 > y > -3.2:
        goal_ok = False

    if 4 > x > 2.5 and 0.7 > y > -3.2:
        goal_ok = False

    if 6.2 > x > 3.8 and -3.3 > y > -4.2:
        goal_ok = False

    if 4.2 > x > 1.3 and 3.7 > y > 1.5:
        goal_ok = False

    if -3.0 > x > -7.2 and 0.5 > y > -1.5:
        goal_ok = False

    if x > 4.5 or x < -4.5 or y > 4.5 or y < -4.5:
        goal_ok = False

    return goal_ok


class GazeboEnv:
    """Superclass for all Gazebo environments."""

    def __init__(self, launchfile, environment_dim, agent_num):
        self.environment_dim = environment_dim

        # 创建一个字典，对每个机器人分别进行初始化
        self.agents = {}
        for i in range(1, agent_num+1):
            agent_name = f"r{i}"
            self.agents[agent_name] = self.create_agent(agent_name)
            print(f"初始化第{i}个agent,{agent_name}")

        print("Roscore launched!")
        rospy.init_node("gym", anonymous=True)
        self.set_state = rospy.Publisher(
                "gazebo/set_model_state", ModelState, queue_size=10
            )
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)

    def create_agent(self , agent_name):
            # 创建一个代理的方法，对每个机器人进行初始化

            agent_model_state = ModelState()
            agent_model_state.model_name = agent_name
            agent_model_state.pose.position.x = 0.0
            agent_model_state.pose.position.y = 0.0
            agent_model_state.pose.position.z = 0.0
            agent_model_state.pose.orientation.x = 0.0
            agent_model_state.pose.orientation.y = 0.0
            agent_model_state.pose.orientation.z = 0.0
            agent_model_state.pose.orientation.w = 1.0

            agent = {
                "model_state": agent_model_state,
                "vel_pub":rospy.Publisher(f"/{agent_name}/cmd_vel" , Twist , queue_size= 1),
                "odom_sub": rospy.Subscriber(f"/{agent_name}/odom", Odometry, self.odom_callback , queue_size=1),
                "velodyne_sub":rospy.Subscriber(f"/{agent_name}/velodyne_points",PointCloud2,self.velodyne_callback,queue_size=1),
                "odom_x": 0,
                "odom_y": 0,
                "goal_x": 1,
                "goal_y": 0.0,
                "upper": 5.0,
                "lower": -5.0,
                "velodyne_data": np.ones(self.environment_dim) * 10,
                "last_odom": None,
                "gaps": self.calculate_gaps()
            }
            agent["model_state"].model_name = agent_name
            return agent
            
    def calculate_gaps(self):
        # 计算和初始化角度间隔
            gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / self.environment_dim]]
            for m in range(self.environment_dim - 1):
                gaps.append([gaps[m][1], gaps[m][1] + np.pi / self.environment_dim])
            gaps[-1][-1] += 0.03
            return gaps

        
    # Read velodyne pointcloud and turn it into distance data, then select the minimum value for each angle
    # range as state representation
    def velodyne_callback(self, v, agent_name):
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        velodyne_data = np.ones(self.environment_dim) * 10
        for i in range(len(data)):
            if data[i][2] > -0.2:
                dot = data[i][0] * 1 + data[i][1] * 0
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)

                for j in range(len(self.agents[agent_name]['gaps'])):
                    if self.agents[agent_name]['gaps'][j][0] <= beta < self.agents[agent_name]['gaps'][j][1]:
                        velodyne_data[j] = min(velodyne_data[j], dist)
                        break
        
        self.agents[agent_name]["velodyne_data"] = velodyne_data
    

    def odom_callback(self, od_data,agent_name):
        self.agents[agent_name]['last_odom'] = od_data

    # Perform an action and read a new state
    def step(self, actions):#todo actions设置为数组
        if len(actions) != len(self.agents):
            print("动作长度不匹配")
            
        targets = [False] * len(self.agents)
        laser_states = {}
        robot_states = {}
        rewards = {}    
        done_list = []

        for i,agent_name in enumerate(self.agents.keys()):
            action = actions[i]
            # Publish the robot action
            vel_cmd = Twist()
            vel_cmd.linear.x = action[0]
            vel_cmd.angular.z = action[1]
            self.agents[agent_name]["vel_pub"].publish(vel_cmd)
            self.publish_markers(action, agent_name)

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        # propagate state for TIME_DELTA seconds
        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            pass
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        for i, agent_name in enumerate(self.agents.keys()):
             # read velodyne laser state
            done_, collision, min_laser = self.observe_collision(self.agents[agent_name]["velodyne_data"])
            v_state = []
            v_state[:] = self.agents[agent_name]["velodyne_data"][:]
            laser_states[agent_name] = [v_state]
       

            # Calculate robot heading from odometry data
            self.agents[agent_name]["odom_x"] = self.agents[agent_name]["last_odom"].pose.pose.position.x
            self.agents[agent_name]["odom_y"] = self.agents[agent_name]["last_odom"].pose.pose.position.y
            quaternion = Quaternion(
                self.agents[agent_name]["last_odom"].pose.pose.orientation.w,
                self.agents[agent_name]["last_odom"].pose.pose.orientation.x,
                self.agents[agent_name]["last_odom"].pose.pose.orientation.y,
                self.agents[agent_name]["last_odom"].pose.pose.orientation.z,
            )
            euler = quaternion.to_euler(degrees=False)
            angle = round(euler[2], 4)

            # Calculate distance to the goal from the robot
            distance = np.linalg.norm(
                [self.agents[agent_name]["odom_x"] - self.agents[agent_name]["goal_x"], self.agents[agent_name]["odom_y"] - self.agents[agent_name]["goal_y"]]
            )

            # Calculate the relative angle between the robots heading and heading toward the goal
            skew_x = self.agents[agent_name]["goal_x"] - self.agents[agent_name]["odom_x"]
            skew_y = self.agents[agent_name]["goal_y"] - self.agents[agent_name]["odom_y"]
            dot = skew_x * 1 + skew_y * 0
            mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
            mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
            beta = math.acos(dot / (mag1 * mag2))
            if skew_y < 0:
                if skew_x < 0:
                    beta = -beta
                else:
                    beta = 0 - beta
            theta = beta - angle
            if theta > np.pi:
                theta = np.pi - theta
                theta = -np.pi - theta
            if theta < -np.pi:
                theta = -np.pi - theta
                theta = np.pi - theta

            # Detect if the goal has been reached and give a large positive reward
            if distance < GOAL_REACHED_DIST:
                targets[i] = True
                done_ = True
            else:
                done_ =False
            done_list.append(done_)
            
            robot_states[agent_name] = [distance, theta, action[0], action[1]]
            rewards[agent_name] = self.get_reward(targets[i], collision, action, min_laser)
        
        done = all(done_list)

        state = np.array(list(laser_states.values())+list(robot_states.values()))
        reward = list(rewards.values())
        return state, reward, done, targets

    def reset(self, start_x=None, start_y=None, start_angle=None):#todo

        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()

        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        for i, agent_name in enumerate(self.agents.keys()):
            # 自定义修改
            angle = start_angle if start_angle is not None else np.random.uniform(-np.pi, np.pi)
            self.agents[agent_name]["angle"] = angle
            quaternion = Quaternion.from_euler(0.0, 0.0, angle)

            # Set position
            if i == 0:  # 第一个机器人
                if start_x is not None and start_y is not None:
                    x = start_x
                    y = start_y
                else:
                    position_ok = False
                    while not position_ok:
                        x = np.random.uniform(-4.5, 4.5)
                        y = np.random.uniform(-4.5, 4.5)
                        position_ok = check_pos(x, y)
            else:  # 其他机器人
                # 使用编队位置函数确定位置
                position_ok = False
                while not position_ok:
                    x_, y_= self.formation(self.agents["r1"]["odom_x"], self.agents["r1"]["odom_y"], i)
                    x = x_+np.random.uniform(-4.5,4.5)
                    y = y_+np.random.uniform(-4.5,4.5)
                    position_ok = check_pos(x,y)

            # 设置机器人状态
            object_state = ModelState()
            object_state.model_name = agent_name
            object_state.pose.position.x = x
            object_state.pose.position.y = y
            object_state.pose.orientation.x = quaternion.x
            object_state.pose.orientation.y = quaternion.y
            object_state.pose.orientation.z = quaternion.z
            object_state.pose.orientation.w = quaternion.w
            self.set_state.publish(object_state)

            # 更新内部状态
            self.agents[agent_name]["odom_x"] = x
            self.agents[agent_name]["odom_y"] = y

        # set a random goal in empty space in environment
        self.change_goal()
        # randomly scatter boxes in the environment
        self.random_box()
        self.publish_markers([0.0, 0.0])

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")
        
        all_robot_states = []
        for agent_name in self.agents.keys():
            v_state = self.agents[agent_name]["velodyne_data"][:]
            laser_state = [v_state]

            distance = np.linalg.norm(
                [self.agents[agent_name]["odom_x"] - self.agents[agent_name]["goal_x"], 
                self.agents[agent_name]["odom_y"] - self.agents[agent_name]["goal_y"]]
            )

            skew_x = self.agents[agent_name]["goal_x"] - self.agents[agent_name]["odom_x"]
            skew_y = self.agents[agent_name]["goal_y"] - self.agents[agent_name]["odom_y"]

            dot = skew_x * 1 + skew_y * 0
            mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
            mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
            beta = math.acos(dot / (mag1 * mag2))

            if skew_y < 0:
                if skew_x < 0:
                    beta = -beta
                else:
                    beta = 0 - beta
            theta = beta - self.agents[agent_name]["angle"]  # 使用每个机器人的角度

            if theta > np.pi:
                theta = np.pi - theta
                theta = -np.pi - theta
            if theta < -np.pi:
                theta = -np.pi - theta
                theta = np.pi - theta

            robot_state = [distance, theta, 0.0, 0.0]
            state = np.append(laser_state, robot_state)
            all_robot_states.append(state)

        # 聚合所有机器人的状态
        states = np.concatenate(all_robot_states)
        return states

    def change_goal(self, goal_x=None, goal_y=None):
        # 为每个机器人设置目标位置
        for i, agent_name in enumerate(self.agents.keys()):
            if i == 0:  # 第一个机器人
                if goal_x is not None and goal_y is not None:
                    self.agents[agent_name]["goal_x"] = goal_x
                    self.agents[agent_name]["goal_y"] = goal_y
                else:
                    # 设置随机目标位置
                    if self.agents["r1"]["upper"] < 10:
                        self.agents["r1"]["upper"] += 0.004
                    if self.agents["r1"]["lower"] > -10:
                        self.agents["r1"]["lower"] -= 0.004
                    goal_ok = False
                    while not goal_ok:
                        self.agents[agent_name]["goal_x"] = self.agents["r1"]["odom_x"] = random.uniform(self.agents["r1"]["upper"],self.agents["r1"]["lower"])
                        self.agents[agent_name]["goal_y"] = self.agents["r1"]["odom_y"] = random.uniform(self.agents["r1"]["upper"],self.agents["r1"]["lower"])
                        goal_ok = check_pos(self.agents[agent_name]["goal_x"], self.agents[agent_name]["goal_y"])
            else:  # 其他机器人
                # 使用编队函数确定位置
                self.agents[agent_name]["goal_x"], self.agents[agent_name]["goal_y"] = self.formation(self.agents["r1"]["goal_x"], self.agents["r1"]["goal_y"], i)

    def formation(leader_x, leader_y, index):
        # 根据具体编队规则来实现
        # 简单的线性编队
        offset = 1.0  # 编队中每个机器人之间的距离
        return leader_x + offset * index, leader_y

    def random_box(self):
        # Randomly change the location of the boxes in the environment on each reset to randomize the training
        # environment
        for i in range(4):
            name = "cardboard_box_" + str(i)

            x = 0
            y = 0
            box_ok = False
            while not box_ok:
                x = np.random.uniform(-6, 6)
                y = np.random.uniform(-6, 6)
                box_ok = check_pos(x, y)
                distance_to_robot = np.linalg.norm([x - self.odom_x, y - self.odom_y])
                distance_to_goal = np.linalg.norm([x - self.goal_x, y - self.goal_y])
                if distance_to_robot < 1.5 or distance_to_goal < 1.5:
                    box_ok = False
            box_state = ModelState()
            box_state.model_name = name
            box_state.pose.position.x = x
            box_state.pose.position.y = y
            box_state.pose.position.z = 0.0
            box_state.pose.orientation.x = 0.0
            box_state.pose.orientation.y = 0.0
            box_state.pose.orientation.z = 0.0
            box_state.pose.orientation.w = 1.0
            self.set_state.publish(box_state)

    # def publish_markers(self, action):
    #     # Publish visual data in Rviz
    #     markerArray = MarkerArray()
    #     marker = Marker()
    #     marker.header.frame_id = "odom"
    #     marker.type = marker.CYLINDER
    #     marker.action = marker.ADD
    #     marker.scale.x = 0.1
    #     marker.scale.y = 0.1
    #     marker.scale.z = 0.01
    #     marker.color.a = 1.0
    #     marker.color.r = 0.0
    #     marker.color.g = 1.0
    #     marker.color.b = 0.0
    #     marker.pose.orientation.w = 1.0
    #     marker.pose.position.x = self.goal_x
    #     marker.pose.position.y = self.goal_y
    #     marker.pose.position.z = 0

    #     markerArray.markers.append(marker)

    #     self.publisher.publish(markerArray)

    #     markerArray2 = MarkerArray()
    #     marker2 = Marker()
    #     marker2.header.frame_id = "odom"
    #     marker2.type = marker.CUBE
    #     marker2.action = marker.ADD
    #     marker2.scale.x = abs(action[0])
    #     marker2.scale.y = 0.1
    #     marker2.scale.z = 0.01
    #     marker2.color.a = 1.0
    #     marker2.color.r = 1.0
    #     marker2.color.g = 0.0
    #     marker2.color.b = 0.0
    #     marker2.pose.orientation.w = 1.0
    #     marker2.pose.position.x = 5
    #     marker2.pose.position.y = 0
    #     marker2.pose.position.z = 0

    #     markerArray2.markers.append(marker2)
    #     self.publisher2.publish(markerArray2)

    #     markerArray3 = MarkerArray()
    #     marker3 = Marker()
    #     marker3.header.frame_id = "odom"
    #     marker3.type = marker.CUBE
    #     marker3.action = marker.ADD
    #     marker3.scale.x = abs(action[1])
    #     marker3.scale.y = 0.1
    #     marker3.scale.z = 0.01
    #     marker3.color.a = 1.0
    #     marker3.color.r = 1.0
    #     marker3.color.g = 0.0
    #     marker3.color.b = 0.0
    #     marker3.pose.orientation.w = 1.0
    #     marker3.pose.position.x = 5
    #     marker3.pose.position.y = 0.2
    #     marker3.pose.position.z = 0

    #     markerArray3.markers.append(marker3)
    #     self.publisher3.publish(markerArray3)

    @staticmethod
    def observe_collision(laser_data):
        # Detect a collision from laser data
        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST:
            return True, True, min_laser
        return False, False, min_laser

    @staticmethod
    def get_reward(target, collision, action, min_laser):
        if target:
            return 100.0
        elif collision:
            return -100.0
        else:
            r3 = lambda x: 1 - x if x < 1 else 0.0
            return action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2
