from collections import deque
import os
import random
from tqdm import tqdm

import torch

from utils_drl import Agent
from utils_env import MyEnv
from utils_memory import ReplayMemory


GAMMA = 0.99
GLOBAL_SEED = 0
MEM_SIZE = 100_000
RENDER = False
SAVE_PREFIX = "./models" # 模型保存地址
STACK_SIZE = 4

EPS_START = 1.
EPS_END = 0.1
EPS_DECAY = 10000 # 1000000

BATCH_SIZE = 32
POLICY_UPDATE = 32
TARGET_UPDATE = 10_000
WARM_STEPS = 50_000
MAX_STEPS = 500_000 # 50000000
EVALUATE_FREQ = 100_00 # 100000

rand = random.Random()
rand.seed(GLOBAL_SEED)
new_seed = lambda: rand.randint(0, 1000_000)
os.mkdir(SAVE_PREFIX) # 创建目录保存模型

torch.manual_seed(new_seed())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
env = MyEnv(device) # 环境
agent = Agent( # 智能体
    env.get_action_dim(), # 3
    device, # cuda
    GAMMA, # 0.99
    new_seed(),
    EPS_START, # 1
    EPS_END, # 0.1
    EPS_DECAY, # 1e6
)
memory = ReplayMemory(STACK_SIZE + 1, MEM_SIZE, device) # 初始化经验池

#### Training ####
obs_queue: deque = deque(maxlen=5)
done = True

progressive = tqdm(range(MAX_STEPS), total = MAX_STEPS, ncols = 50, leave = False, unit="b") # 可视化进度条
for step in progressive:
    if done: # 开始新一轮环境
        observations, _, _ = env.reset()
        for obs in observations: obs_queue.append(obs)

    training = len(memory) > WARM_STEPS # len(memory) = step，50000，先存储一些数据用于后续训练
    state = env.make_state(obs_queue).to(device).float() # 丢掉第一个状态(~)
    action = agent.run(state, training) # 选取动作a（以eps的概率随机选取动作a, 否则a由Q网络选取）
    obs, reward, done = env.step(action) # 执行动作a, 获取奖励reward和下一个状态s’
    obs_queue.append(obs) # 加入新的状态
    memory.push(env.make_folded_state(obs_queue), action, reward, done) # 储存（s,a,r）到经验池中
    if step % POLICY_UPDATE == 0 and training: agent.learn(memory, BATCH_SIZE) # 训练
    if step % TARGET_UPDATE == 0: agent.sync() # 将策略网络中的权重同步到目标网络
    if step % EVALUATE_FREQ == 0:
        avg_reward, frames = env.evaluate(obs_queue, agent, render = RENDER) # 让几个AI玩一下游戏，记录一下奖励
        with open("rewards.txt", "a") as fp: # 文件记录奖励，用于后续画图
            fp.write(f"{step//EVALUATE_FREQ:3d} {step:8d} {avg_reward:.1f}\n")
        if RENDER: # No～
            prefix = f"eval_{step//EVALUATE_FREQ:03d}"
            os.mkdir(prefix)
            for ind, frame in enumerate(frames):
                with open(os.path.join(prefix, f"{ind:06d}.png"), "wb") as fp:
                    frame.save(fp, format="png")
        agent.save(os.path.join(SAVE_PREFIX, f"model_{step//EVALUATE_FREQ:03d}")) # 保存模型
        done = True

