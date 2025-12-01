import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter
from collections import deque

from replay_buffer import ReplayBuffer
from velodyne_env import GazeboEnv

class ExplorationGrid:
    """探索网格映射系统，用于跟踪环境探索覆盖率"""
    
    def __init__(self, grid_size=200, resolution=0.1):  # 20m×20m地图，分辨率0.1m
        self.grid_size = grid_size
        self.resolution = resolution
        self.grid = np.zeros((grid_size, grid_size), dtype=bool)  # 是否被探索
        self.visited_cells = 0
        self.total_cells = grid_size * grid_size
        self.coverage_threshold = 0.85  # 85%覆盖率目标
        self.origin_x = grid_size // 2 * resolution  # 地图原点x
        self.origin_y = grid_size // 2 * resolution  # 地图原点y
    
    def update(self, x, y):
        """更新机器人位置到探索网格"""
        i = int(x / self.resolution) + self.grid_size // 2
        j = int(y / self.resolution) + self.grid_size // 2
        
        if 0 <= i < self.grid_size and 0 <= j < self.grid_size:
            if not self.grid[i, j]:
                self.grid[i, j] = True
                self.visited_cells += 1
                return True  # 新探索的单元格
        return False
    
    def get_coverage_ratio(self):
        """获取当前覆盖率比例"""
        return self.visited_cells / self.total_cells
    
    def reset(self):
        """重置探索网格"""
        self.grid.fill(False)
        self.visited_cells = 0

class CuriosityDrivenExploration:
    """好奇心驱动探索模块，鼓励探索新状态"""
    
    def __init__(self, state_dim, memory_size=1000, novelty_bonus=0.1):
        self.visited_states = deque(maxlen=memory_size)
        self.novelty_bonus = novelty_bonus
        self.state_dim = state_dim
    
    def compute_novelty_reward(self, state):
        """计算状态新颖性奖励"""
        # 提取位置和传感器特征用于新颖性判断
        state_features = state[:min(8, len(state))]  # 使用前8个特征计算新颖性
        
        if len(self.visited_states) < 10:
            self.visited_states.append(state_features.copy())
            return self.novelty_bonus
        
        # 计算与最近邻状态的差异
        distances = [np.linalg.norm(state_features - visited) 
                    for visited in list(self.visited_states)[-100:]]
        min_distance = min(distances) if distances else 0
        
        if min_distance > 0.3:  # 新状态奖励
            self.visited_states.append(state_features.copy())
            return self.novelty_bonus
        return 0

def calculate_reward(state, next_state, done, min_laser_distance, coverage_ratio, 
                    step_count, max_steps_per_episode, curiosity_reward=0):
    """计算综合奖励函数"""
    reward = 0
    
    # 1. 碰撞惩罚（严重惩罚）
    if min_laser_distance < 0.1:  # 10cm内视为碰撞
        reward -= 10
        print("Collision detected! Reward -10")
        return reward, True  # 返回奖励和done标志
    
    # 2. 靠近障碍物惩罚
    elif min_laser_distance < 0.2:  # 20cm内靠近障碍物
        reward -= 1
        print("Near obstacle! Reward -1")
    
    # 3. 探索覆盖率奖励/惩罚
    if coverage_ratio >= 0.85:
        reward += 1
        print(f"Coverage {coverage_ratio:.1%} >= 85%! Reward +1")
    else:
        reward -= 0.5  # 适度惩罚，避免过于严厉
    
    # 4. 进度奖励（鼓励移动和探索）
    if not done and min_laser_distance > 0.3:  # 安全距离内移动
        # 基于移动距离的小幅度奖励
        if len(state) >= 2 and len(next_state) >= 2:
            movement_reward = 0.05
            reward += movement_reward
    
    # 5. 好奇心驱动探索奖励
    reward += curiosity_reward
    
    # 6. 时间步惩罚（鼓励效率）
    if step_count > max_steps_per_episode * 0.8:  # 后20%时间步
        reward -= 0.01
    
    return reward, False

def evaluate(network, epoch, eval_episodes=10):
    """评估函数 - 修改为使用新的奖励逻辑"""
    avg_reward = 0.0
    avg_coverage = 0.0
    success_count = 0
    
    for _ in range(eval_episodes):
        episode_reward = 0
        state = env.reset()
        exploration_grid.reset()
        curiosity_module = CuriosityDrivenExploration(len(state))
        done = False
        step_count = 0
        max_steps = 500
        
        while not done and step_count < max_steps:
            action = network.get_action(np.array(state))
            a_in = [(action[0] + 1) / 2, action[1]]  # 转换动作范围
            
            next_state, _, env_done, _ = env.step(a_in)
            
            # 获取激光雷达数据（20维360度）
            laser_data = next_state[4:24] if len(next_state) > 24 else next_state[:20]
            min_laser_distance = min(laser_data) if len(laser_data) > 0 else float('inf')
            
            # 更新探索网格（使用里程计信息）
            if hasattr(env, 'odom_x') and hasattr(env, 'odom_y'):
                exploration_grid.update(env.odom_x, env.odom_y)
            
            coverage_ratio = exploration_grid.get_coverage_ratio()
            curiosity_reward = curiosity_module.compute_novelty_reward(next_state)
            
            # 计算自定义奖励
            custom_reward, collision_done = calculate_reward(
                state, next_state, env_done, min_laser_distance, 
                coverage_ratio, step_count, max_steps, curiosity_reward
            )
            
            done = env_done or collision_done
            episode_reward += custom_reward
            state = next_state
            step_count += 1
            
            if coverage_ratio >= 0.85:
                success_count += 1
                break
        
        avg_reward += episode_reward
        avg_coverage += coverage_ratio
    
    avg_reward /= eval_episodes
    avg_coverage /= eval_episodes
    success_rate = success_count / eval_episodes
    
    print("..............................................")
    print(f"Epoch {epoch} Evaluation Results:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Coverage: {avg_coverage:.1%}")
    print(f"Success Rate (85%+ coverage): {success_rate:.1%}")
    print("..............................................")
    
    return avg_reward, avg_coverage, success_rate

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # 第一个Critic网络
        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800 + action_dim, 600)
        self.layer_3 = nn.Linear(600, 1)
        
        # 第二个Critic网络
        self.layer_4 = nn.Linear(state_dim, 800)
        self.layer_5 = nn.Linear(800 + action_dim, 600)
        self.layer_6 = nn.Linear(600, 1)

    def forward(self, s, a):
        # 第一个Q网络
        s1 = F.relu(self.layer_1(s))
        s1_a = torch.cat([s1, a], 1)
        q1 = F.relu(self.layer_2(s1_a))
        q1 = self.layer_3(q1)
        
        # 第二个Q网络
        s2 = F.relu(self.layer_4(s))
        s2_a = torch.cat([s2, a], 1)
        q2 = F.relu(self.layer_5(s2_a))
        q2 = self.layer_6(q2)
        
        return q1, q2

class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        # 初始化Actor网络
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        
        # 初始化Critic网络
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        
        self.max_action = max_action
        self.writer = SummaryWriter()
        self.iter_count = 0
        self.convergence_threshold = 100
        self.recent_rewards = deque(maxlen=100)

    def get_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, 
               tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        
        av_Q = 0
        max_Q = -inf
        av_loss = 0
        
        for it in range(iterations):
            # 从回放缓冲区采样
            batch_states, batch_actions, batch_rewards, batch_dones, batch_next_states = replay_buffer.sample_batch(batch_size)
            
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)

            # 计算目标Q值
            with torch.no_grad():
                next_action = self.actor_target(next_state)
                noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
                noise = noise.clamp(-noise_clip, noise_clip)
                next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
                
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + ((1 - done) * discount * target_Q)

            # 更新Critic网络
            current_Q1, current_Q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # 延迟策略更新
            if it % policy_freq == 0:
                # 更新Actor网络
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # 软更新目标网络
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            av_loss += critic_loss.item()
            av_Q += torch.mean(target_Q).item()
            max_Q = max(max_Q, torch.max(target_Q).item())

        self.iter_count += 1
        # 记录训练指标
        self.writer.add_scalar("loss", av_loss / iterations, self.iter_count)
        self.writer.add_scalar("Av. Q", av_Q / iterations, self.iter_count)
        self.writer.add_scalar("Max. Q", max_Q, self.iter_count)
        
        return av_loss / iterations

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load("%s/%s_actor.pth" % (directory, filename)))
        self.critic.load_state_dict(torch.load("%s/%s_critic.pth" % (directory, filename)))

# 设置训练参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 0
eval_freq = 5000  # 评估频率
max_ep = 500  # 每回合最大步数
eval_ep = 10  # 评估回合数
max_timesteps = 500000  # 最大训练步数
expl_noise = 1.0  # 初始探索噪声
expl_decay_steps = 200000  # 探索衰减步数
expl_min = 0.1  # 最小探索噪声
batch_size = 100  # 批次大小
discount = 0.99  # 折扣因子
tau = 0.005  # 软更新参数
policy_noise = 0.2  # 策略噪声
noise_clip = 0.5  # 噪声裁剪
policy_freq = 2  # 策略更新频率
buffer_size = 1000000  # 回放缓冲区大小
file_name = "TD3_autonomous_exploration"
save_model = True
load_model = False
random_near_obstacle = True

# 创建存储目录
if not os.path.exists("./results"):
    os.makedirs("./results")
if save_model and not os.path.exists("./pytorch_models"):
    os.makedirs("./pytorch_models")

# 创建环境和网络
environment_dim = 20
robot_dim = 4
env = GazeboEnv("multi_robot_scenario.launch", environment_dim)
time.sleep(5)

torch.manual_seed(seed)
np.random.seed(seed)

state_dim = environment_dim + robot_dim
action_dim = 2
max_action = 1

# 初始化网络和组件
network = TD3(state_dim, action_dim, max_action)
replay_buffer = ReplayBuffer(buffer_size, seed)
exploration_grid = ExplorationGrid()

# 训练变量初始化
timestep = 0
timesteps_since_eval = 0
episode_num = 0
done = True
epoch = 1
evaluations = []
convergence_count = 0
target_convergence = 100  # 目标收敛奖励值
convergence_threshold = 5  # 收敛阈值

print("开始自主探索训练...")
print(f"目标收敛奖励: {target_convergence} ± {convergence_threshold}")

# 主训练循环
while timestep < max_timesteps:
    if done:
        if timestep != 0:
            # 训练网络
            avg_loss = network.train(
                replay_buffer,
                episode_timesteps,
                batch_size,
                discount,
                tau,
                policy_noise,
                noise_clip,
                policy_freq,
            )
            
            print(f"回合 {episode_num} 完成, 总步数: {timestep}, 平均损失: {avg_loss:.4f}")

        # 定期评估
        if timesteps_since_eval >= eval_freq:
            timesteps_since_eval %= eval_freq
            avg_reward, avg_coverage, success_rate = evaluate(
                network=network, epoch=epoch, eval_episodes=eval_ep
            )
            
            evaluations.append((avg_reward, avg_coverage, success_rate))
            
            # 保存模型
            if save_model:
                filename = f"{file_name}_epoch_{epoch}_reward_{avg_reward:.2f}"
                network.save(filename, directory="./pytorch_models")
            
            # 检查收敛
            if abs(avg_reward - target_convergence) < convergence_threshold:
                convergence_count += 1
                print(f"接近收敛目标! ({convergence_count}/5)")
                if convergence_count >= 5:
                    print(f"训练在奖励值 {avg_reward:.2f} 处收敛!")
                    break
            else:
                convergence_count = 0
            
            epoch += 1

        # 重置环境
        state = env.reset()
        exploration_grid.reset()
        curiosity_module = CuriosityDrivenExploration(state_dim)
        
        done = False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

    # 衰减探索噪声
    if expl_noise > expl_min:
        expl_noise -= (1 - expl_min) / expl_decay_steps

    # 选择动作并添加探索噪声
    action = network.get_action(np.array(state))
    action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(
        -max_action, max_action
    )

    # 靠近障碍物时的随机探索
    if random_near_obstacle:
        laser_data = state[4:24] if len(state) > 24 else state[:20]
        min_laser = min(laser_data) if len(laser_data) > 0 else float('inf')
        
        if (np.random.uniform(0, 1) > 0.8 and min_laser < 0.6 and 
            hasattr(env, 'count_rand_actions') and env.count_rand_actions < 1):
            env.count_rand_actions = np.random.randint(5, 10)
            env.random_action = np.random.uniform(-1, 1, 2)

        if hasattr(env, 'count_rand_actions') and env.count_rand_actions > 0:
            env.count_rand_actions -= 1
            action = env.random_action
            action[0] = -1  # 强制后退

    # 执行动作
    a_in = [(action[0] + 1) / 2, action[1]]
    next_state, _, env_done, _ = env.step(a_in)

    # 获取激光雷达数据
    laser_data = next_state[4:24] if len(next_state) > 24 else next_state[:20]
    min_laser_distance = min(laser_data) if len(laser_data) > 0 else float('inf')
    
    # 更新探索网格
    if hasattr(env, 'odom_x') and hasattr(env, 'odom_y'):
        exploration_grid.update(env.odom_x, env.odom_y)
    
    coverage_ratio = exploration_grid.get_coverage_ratio()
    curiosity_reward = curiosity_module.compute_novelty_reward(next_state)
    
    # 计算自定义奖励
    custom_reward, collision_done = calculate_reward(
        state, next_state, env_done, min_laser_distance, 
        coverage_ratio, episode_timesteps, max_ep, curiosity_reward
    )
    
    done = env_done or collision_done or (episode_timesteps + 1 >= max_ep)
    done_bool = 0 if episode_timesteps + 1 == max_ep else int(done)
    
    # 存储经验
    replay_buffer.add(state, action, custom_reward, done_bool, next_state)
    
    # 更新状态和计数器
    state = next_state
    episode_reward += custom_reward
    episode_timesteps += 1
    timestep += 1
    timesteps_since_eval += 1
    
    # 定期打印进度
    if timestep % 1000 == 0:
        print(f"步数: {timestep}, 覆盖率: {coverage_ratio:.1%}, 回合奖励: {episode_reward:.2f}")

# 训练结束后的最终评估
print("训练完成，进行最终评估...")
final_reward, final_coverage, final_success = evaluate(
    network=network, epoch=epoch, eval_episodes=eval_ep
)

print(f"\n=== 最终训练结果 ===")
print(f"最终平均奖励: {final_reward:.2f}")
print(f"最终平均覆盖率: {final_coverage:.1%}")
print(f"最终成功率: {final_success:.1%}")

if save_model:
    network.save(f"{file_name}_final", directory="./pytorch_models")
    np.save("./results/evaluations.npy", evaluations)

print("自主探索回合训练完成!")