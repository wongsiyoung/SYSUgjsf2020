from typing import (
    Tuple,
)

import torch

from utils_types import (
    BatchAction,
    BatchDone,
    BatchNext,
    BatchReward,
    BatchState,
    TensorStack5,
    TorchDevice,
)


class ReplayMemory(object): # 用于存储游戏的中间样本（经验池）

    def __init__(
            self,
            channels: int, # 5
            capacity: int, # 1e5
            device: TorchDevice,
    ) -> None:
        self.__device = device # cuda
        self.__capacity = capacity # 1e5
        self.__size = 0
        self.__pos = 0
        # 四块经验池：状态、动作、收益、是否完成
        self.__m_states = torch.zeros((capacity, channels, 84, 84), dtype=torch.uint8) # 状态
        self.__m_actions = torch.zeros((capacity, 1), dtype=torch.long) # 动作
        self.__m_rewards = torch.zeros((capacity, 1), dtype=torch.int8) # 收益
        self.__m_dones = torch.zeros((capacity, 1), dtype=torch.bool) # 是否完成

    def push(
            self,
            folded_state: TensorStack5,
            action: int,
            reward: int,
            done: bool,
    ) -> None: # 保存一次交互：状态、动作、收益、是否完成
        self.__m_states[self.__pos] = folded_state # 状态
        self.__m_actions[self.__pos, 0] = action # 动作
        self.__m_rewards[self.__pos, 0] = reward # 收益
        self.__m_dones[self.__pos, 0] = done # 是否完成
        
        self.__pos = (self.__pos + 1) % self.__capacity # 位置+1
        self.__size = max(self.__size, self.__pos)

    def sample(self, batch_size: int) -> Tuple[
            BatchState,
            BatchAction,
            BatchReward,
            BatchNext,
            BatchDone,
    ]: # 在经验池中选取一组transition样本集（minibatch）
        indices = torch.randint(0, high=self.__size, size=(batch_size,)) # 随机选取经验
        b_state = self.__m_states[indices, :4].to(self.__device).float() # 当前状态：5个中的前4个
        b_next = self.__m_states[indices, 1:].to(self.__device).float() # 下一个状态：5个中的后4个
        b_action = self.__m_actions[indices].to(self.__device) # 动作
        b_reward = self.__m_rewards[indices].to(self.__device).float() # 收益
        b_done = self.__m_dones[indices].to(self.__device).float() # 是否完成
        return b_state, b_action, b_reward, b_next, b_done

    def __len__(self) -> int:
        return self.__size