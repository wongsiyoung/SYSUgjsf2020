from typing import (
    Optional,
)

import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils_types import (
    TensorStack4,
    TorchDevice,
)

from utils_memory import ReplayMemory
from utils_model import DQN


class Agent(object): # 智能体的配置

    def __init__(
            self,
            action_dim: int, # 3
            device: TorchDevice, # cuda
            gamma: float, # 0.99
            seed: int,

            eps_start: float, # 1
            eps_final: float, # 0.1
            eps_decay: float, # 10000

            restore: Optional[str] = None,
    ) -> None:
        self.__action_dim = action_dim # 3
        self.__device = device
        self.__gamma = gamma # 0.99

        self.__eps_start = eps_start # 1
        self.__eps_final = eps_final # 0.1
        self.__eps_decay = eps_decay # 1e6

        self.__eps = eps_start # 1
        self.__r = random.Random()
        self.__r.seed(seed)

        self.__policy = DQN(action_dim, device).to(device) # 策略DQN
        self.__target = DQN(action_dim, device).to(device) # 目标DQN，减少目标计算与当前值的相关性
        
        if restore is None: self.__policy.apply(DQN.init_weights)
        else: self.__policy.load_state_dict(torch.load(restore))
            
        self.__target.load_state_dict(self.__policy.state_dict()) # 将策略网络中的权重同步到目标网络
        self.__optimizer = optim.Adam(self.__policy.parameters(), lr=0.0000625, eps=1.5e-4, ) # 优化器
        self.__target.eval() # 验证模式

    def run(self, state: TensorStack4, training: bool = False) -> int:
        """run suggests an action for the given state."""
        if training: # 修改eps，逐渐降低
            self.__eps -= (self.__eps_start - self.__eps_final) / self.__eps_decay
            self.__eps = max(self.__eps, self.__eps_final)

        if self.__r.random() > self.__eps: # 1-eps概率取最大值
            with torch.no_grad(): 
                return self.__policy(state).max(1).indices.item()
        return self.__r.randint(0, self.__action_dim - 1) # eps概率随机

    def learn(self, memory: ReplayMemory, batch_size: int) -> float: # 训练
        """learn trains the value network via TD-learning."""
        state_batch, action_batch, reward_batch, next_batch, done_batch = memory.sample(batch_size) # 在经验池中选取一组transition样本集（minibatch）
        values = self.__policy(state_batch.float()).gather(1, action_batch) # DQN输出y
        values_next = self.__target(next_batch.float()).max(1).values.detach() # max_a(Q(S',a))
        expected = (self.__gamma * values_next.unsqueeze(1)) * (1. - done_batch) + reward_batch # Q-Learning计算Q(S,A)
        loss = F.smooth_l1_loss(values, expected) # 计算误差
        # 更新网络参数三部曲
        self.__optimizer.zero_grad()
        loss.backward()
        for param in self.__policy.parameters(): param.grad.data.clamp_(-1, 1)
        self.__optimizer.step()

        return loss.item() # 返回误差

    def sync(self) -> None: # 将策略网络中的权重同步到目标网络
        """sync synchronizes the weights from the policy network to the target
        network."""
        self.__target.load_state_dict(self.__policy.state_dict())

    def save(self, path: str) -> None: # 保存模型
        """save saves the state dict of the policy network."""
        torch.save(self.__policy.state_dict(), path)