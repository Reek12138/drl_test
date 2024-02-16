import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter

from replay_buffer import ReplayBuffer
# from velodyne_env import GazeboEnv
from velodyne_env_test2 import GazeboEnv


# def evaluate(network, epoch, eval_episodes=10):
#     avg_reward = 0.0
#     col = 0
#     for _ in range(eval_episodes):
#         count = 0
#         state = env.reset()
#         done = False
#         while not done and count < 501:
#             action = network.get_action(np.array(state))
#             a_in = [(action[0] + 1) / 2, action[1]]
#             state, reward, done, _ = env.step(a_in)
#             avg_reward += reward
#             count += 1
#             if reward < -90:
#                 col += 1
#     avg_reward /= eval_episodes
#     avg_col = col / eval_episodes
#     print("..............................................")
#     print(
#         "Average Reward over %i Evaluation Episodes, Epoch %i: %f, %f"
#         % (eval_episodes, epoch, avg_reward, avg_col)
#     )
#     print("..............................................")
#     return avg_reward
def evaluate(network, epoch, eval_episodes=10):
    num_agents = len(env.agents)  # 获取智能体的数量
    avg_rewards = np.zeros(num_agents)
    collisions = np.zeros(num_agents)

    for _ in range(eval_episodes):
        state = env.reset()  # 重置环境并获取初始状态
        done = False
        count = 0

        while not done and count<801:
            actions = []
            for agent_state in state:  # 假设state是包含所有智能体状态的列表
                action = network.get_actions(np.array(agent_state))  # 获取动作
                a_in = [(action[0] + 1) / 2, action[1]]  # 调整动作格式
                actions.append(a_in)

            next_state, rewards, done, _ = env.step(actions)  # 执行动作并获取下一状态、奖励、是否完成

            count += 1
            avg_rewards += rewards  # 累加每个智能体的奖励
            collisions += [1 if r < -90 else 0 for r in rewards]  # 碰撞判断

            state = next_state

    avg_rewards /= eval_episodes  # 计算平均奖励
    avg_collisions = collisions / eval_episodes  # 计算平均碰撞次数

    # 打印结果
    print("..............................................")
    for i in range(num_agents):
        print(f"Agent {i}: Average Reward over {eval_episodes} Evaluation Episodes, Epoch {epoch}: {avg_rewards[i]}, Collisions: {avg_collisions[i]}")
    print("..............................................")
    return avg_rewards, avg_collisions

# class Actor(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(Actor, self).__init__()

#         self.layer_1 = nn.Linear(state_dim, 800)
#         self.layer_2 = nn.Linear(800, 600)
#         self.layer_3 = nn.Linear(600, action_dim)
#         self.tanh = nn.Tanh()

#     def forward(self, s):
#         s = F.relu(self.layer_1(s))
#         s = F.relu(self.layer_2(s))
#         a = self.tanh(self.layer_3(s))
#         return a


# class Critic(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(Critic, self).__init__()

#         self.layer_1 = nn.Linear(state_dim, 800)
#         self.layer_2_s = nn.Linear(800, 600)
#         self.layer_2_a = nn.Linear(action_dim, 600)
#         self.layer_3 = nn.Linear(600, 1)

#         self.layer_4 = nn.Linear(state_dim, 800)
#         self.layer_5_s = nn.Linear(800, 600)
#         self.layer_5_a = nn.Linear(action_dim, 600)
#         self.layer_6 = nn.Linear(600, 1)

#     def forward(self, s, a):
#         s1 = F.relu(self.layer_1(s))
#         self.layer_2_s(s1)
#         self.layer_2_a(a)
#         s11 = torch.mm(s1, self.layer_2_s.weight.data.t())
#         s12 = torch.mm(a, self.layer_2_a.weight.data.t())
#         s1 = F.relu(s11 + s12 + self.layer_2_a.bias.data)
#         q1 = self.layer_3(s1)

#         s2 = F.relu(self.layer_4(s))
#         self.layer_5_s(s2)
#         self.layer_5_a(a)
#         s21 = torch.mm(s2, self.layer_5_s.weight.data.t())
#         s22 = torch.mm(a, self.layer_5_a.weight.data.t())
#         s2 = F.relu(s21 + s22 + self.layer_5_a.bias.data)
#         q2 = self.layer_6(s2)
#         return q1, q2

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, num_agents):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim * num_agents, 800*num_agents)
        self.layer_2 = nn.Linear(800*num_agents, 300*agent_num)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()


    def forward(self, s, a):
        s1 = F.relu(self.layer_1(s))
        self.layer_2_s(s1)
        self.layer_2_a(a)
        s11 = torch.mm(s1, self.layer_2_s.weight.data.t())
        s12 = torch.mm(a, self.layer_2_a.weight.data.t())
        s1 = F.relu(s11 + s12 + self.layer_2_a.bias.data)
        q1 = self.layer_3(s1)

        s2 = F.relu(self.layer_4(s))
        self.layer_5_s(s2)
        self.layer_5_a(a)
        s21 = torch.mm(s2, self.layer_5_s.weight.data.t())
        s22 = torch.mm(a, self.layer_5_a.weight.data.t())
        s2 = F.relu(s21 + s22 + self.layer_5_a.bias.data)
        q2 = self.layer_6(s2)
        return q1, q2


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, num_agents):
        super(Critic, self).__init__()
        # 输入维度调整为所有智能体的状态和动作之和
        self.layer_1 = nn.Linear(state_dim * num_agents + action_dim * num_agents, 800*num_agents)
        self.layer_2_s = nn.Linear(800*num_agents, 300*num_agents)
        self.layer_2_a = nn.Linear(action_dim*num_agents, 600)
        self.layer_3 = nn.Linear(600, 1)

        self.layer_4 = nn.Linear(state_dim * num_agents + action_dim * num_agents, 800*num_agents)
        self.layer_5_s = nn.Linear(800*num_agents, 300*num_agents)
        self.layer_5_a = nn.Linear(action_dim*num_agents, 600)
        self.layer_6 = nn.Linear(600, 1)

    def forward(self, s, a):
        x = torch.cat((s, a), dim=1)
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        q = self.layer_3(x)
        return q



# TD3 network
class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        # Initialize the Actor network
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        # Initialize the Critic networks
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.max_action = max_action
        self.writer = SummaryWriter()
        self.iter_count = 0

    def get_action(self, state):
        # Function to get the action from the actor
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def get_actions(self, states):
        # Function to get actions for multiple agents
        actions = []
        for state in states:
            state_tensor = torch.Tensor(state.reshape(1, -1)).to(device)
            action = self.actor(state_tensor).cpu().data.numpy().flatten()
            actions.append(action)
        return actions


    # training cycle TODO
    def train(
        self,
        replay_buffer,
        iterations,
        batch_size=100,
        discount=1,
        tau=0.005,
        policy_noise=0.2,  # discount=0.99
        noise_clip=0.5,
        policy_freq=2,
    ):
        av_Q = 0
        max_Q = -inf
        av_loss = 0
        for it in range(iterations):
            # sample a batch from the replay buffer
            (
                batch_states,
                batch_actions,
                batch_rewards,
                batch_dones,
                batch_next_states,
            ) = replay_buffer.sample_batch(batch_size)
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)

            # Obtain the estimated action from the next state by using the actor-target
            next_action = self.actor_target(next_state)

            # Add noise to the action
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Calculate the Q values from the critic-target network for the next state-action pair
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)

            # Select the minimal Q value from the 2 calculated values
            target_Q = torch.min(target_Q1, target_Q2)
            av_Q += torch.mean(target_Q)
            max_Q = max(max_Q, torch.max(target_Q))
            # Calculate the final Q value from the target network parameters by using Bellman equation
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # Get the Q values of the basis networks with the current parameters
            current_Q1, current_Q2 = self.critic(state, action)

            # Calculate the loss between the current Q value and the target Q value
            loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Perform the gradient descent
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()

            if it % policy_freq == 0:
                # Maximize the actor output value by performing gradient descent on negative Q values
                # (essentially perform gradient ascent)
                actor_grad, _ = self.critic(state, self.actor(state))
                actor_grad = -actor_grad.mean()
                self.actor_optimizer.zero_grad()
                actor_grad.backward()
                self.actor_optimizer.step()

                # Use soft update to update the actor-target network parameters by
                # infusing small amount of current parameters
                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )
                # Use soft update to update the critic-target network parameters by infusing
                # small amount of current parameters
                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

            av_loss += loss
        self.iter_count += 1
        # Write new values for tensorboard
        self.writer.add_scalar("loss", av_loss / iterations, self.iter_count)
        self.writer.add_scalar("Av. Q", av_Q / iterations, self.iter_count)
        self.writer.add_scalar("Max. Q", max_Q, self.iter_count)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )
        self.critic.load_state_dict(
            torch.load("%s/%s_critic.pth" % (directory, filename))
        )


# Set the parameters for the implementation
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu
device = torch.device("cpu")  # cuda or cpu
seed = 0  # Random seed number
eval_freq = 5e3  # After how many steps to perform the evaluation
max_ep = 500  # maximum number of steps per episode
eval_ep = 10  # number of episodes for evaluation
max_timesteps = 5e6  # Maximum number of steps to perform
expl_noise = 1  # Initial exploration noise starting value in range [expl_min ... 1]
expl_decay_steps = (
    500000  # Number of steps over which the initial exploration noise will decay over
)
expl_min = 0.1  # Exploration noise after the decay in range [0...expl_noise]
batch_size = 40  # Size of the mini-batch
discount = 0.99999  # Discount factor to calculate the discounted future reward (should be close to 1)
tau = 0.005  # Soft target update variable (should be close to 0)
policy_noise = 0.2  # Added noise for exploration
noise_clip = 0.5  # Maximum clamping values of the noise
policy_freq = 2  # Frequency of Actor network updates
buffer_size = 1e6  # Maximum size of the buffer
file_name = "TD3_velodyne"  # name of the file to store the policy
save_model = True  # Weather to save the model or not
load_model = False  # Weather to load a stored model
random_near_obstacle = True  # To take random actions near obstacles or not

agent_num = 2#机器人数量，需要与launch文件匹配

# Create the network storage folders
if not os.path.exists("./results"):
    os.makedirs("./results")
if save_model and not os.path.exists("./pytorch_models"):
    os.makedirs("./pytorch_models")

# Create the training environment
environment_dim = 20
r1 = "r1"
robot_dim = 4
env = GazeboEnv("multi_robot_scenario.launch", environment_dim , agent_num)
time.sleep(5)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = environment_dim + robot_dim
action_dim = 2
max_action = 1

# Create the network
network = TD3(state_dim, action_dim, max_action)
# Create a replay buffer
replay_buffer = ReplayBuffer(buffer_size, seed)
if load_model:
    try:
        network.load(file_name, "./pytorch_models")
    except:
        print(
            "Could not load the stored model parameters, initializing training with random parameters"
        )

# Create evaluation data store
evaluations = []

timestep = 0
timesteps_since_eval = 0
episode_num = 0
done = True
epoch = 1

count_rand_actions = 0
random_action = []

# Begin the training loop
while timestep < max_timesteps:

    # On termination of episode
    if done:
        if timestep != 0:
            network.train(
                replay_buffer,
                episode_timesteps,
                batch_size,
                discount,
                tau,
                policy_noise,
                noise_clip,
                policy_freq,
            )

        if timesteps_since_eval >= eval_freq:
            print("Validating")
            timesteps_since_eval %= eval_freq
            evaluations.append(
                evaluate(network=network, epoch=epoch, eval_episodes=eval_ep)
            )
            network.save(file_name, directory="./pytorch_models")
            np.save("./results/%s" % (file_name), evaluations)
            epoch += 1

        state = env.reset()
        done = False

        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

    # add some exploration noise
    if expl_noise > expl_min:
        expl_noise = expl_noise - ((1 - expl_min) / expl_decay_steps)

    action = network.get_action(np.array(state))
    action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(
        -max_action, max_action
    )

    # If the robot is facing an obstacle, randomly force it to take a consistent random action.
    # This is done to increase exploration in situations near obstacles.
    # Training can also be performed without it
    if random_near_obstacle:
        if (
            np.random.uniform(0, 1) > 0.85
            and min(state[4:-8]) < 0.6
            and count_rand_actions < 1
        ):
            count_rand_actions = np.random.randint(8, 15)
            random_action = np.random.uniform(-1, 1, 2)

        if count_rand_actions > 0:
            count_rand_actions -= 1
            action = random_action
            action[0] = -1

    # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity
    a_in = [(action[0] + 1) / 2, action[1]]
    next_state, reward, done, target = env.step(a_in)
    done_bool = 0 if episode_timesteps + 1 == max_ep else int(done)
    done = 1 if episode_timesteps + 1 == max_ep else int(done)
    episode_reward += reward

    # Save the tuple in replay buffer
    replay_buffer.add(state, action, reward, done_bool, next_state)

    # Update the counters
    state = next_state
    episode_timesteps += 1
    timestep += 1
    timesteps_since_eval += 1

# After the training is done, evaluate the network and save it
evaluations.append(evaluate(network=network, epoch=epoch, eval_episodes=eval_ep))
if save_model:
    network.save("%s" % file_name, directory="./models")
np.save("./results/%s" % file_name, evaluations)
