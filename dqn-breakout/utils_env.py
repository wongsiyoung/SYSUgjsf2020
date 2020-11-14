from typing import (
    List,
    Optional,
    Tuple,
)

import base64
from collections import deque
import pathlib

from IPython import display as ipydisplay
import numpy as np
from PIL import Image
import torch

from vendor.atari_wrappers import make_atari, wrap_deepmind
from utils_types import (
    GymImg,
    GymObs,
    TensorObs,
    TensorStack4,
    TensorStack5,
    TorchDevice,
)
from utils_drl import Agent

# HTML_TEMPLATE：一个用于显示mp4视频的模板元素
HTML_TEMPLATE = """<video alt="{alt}" autoplay loop controls style="height: 400px;">
  <source src="data:video/mp4;base64,{data}" type="video/mp4" />
</video>"""

class MyEnv(object): # 自定义的游戏环境

    def __init__(self, device: TorchDevice) -> None:
        env_raw = make_atari("BreakoutNoFrameskip-v4") # 加载预有的breakout模型
        self.__env_train = wrap_deepmind(env_raw, episode_life=True) # 训练环境
        env_raw = make_atari("BreakoutNoFrameskip-v4") # 加载预有的breakout模型
        self.__env_eval = wrap_deepmind(env_raw, episode_life=True) # 测试环境
        self.__env = self.__env_train # 默认为训练环境
        self.__device = device

    def reset(self, render: bool = False, ) -> Tuple[List[TensorObs], float, List[GymImg]]: # 重置并初始化底层环境
        """reset resets and initializes the underlying gym environment."""
        self.__env.reset()
        init_reward = 0.
        observations = []
        frames = []
        for _ in range(5): # no-op
            obs, reward, done = self.step(0) # step函数返回下一状态、收益、是否结束
            observations.append(obs)
            init_reward += reward # 统计初始收益
            if done: return self.reset(render)
            if render: # 返回帧（默认False）
                frames.append(self.get_frame())

        return observations, init_reward, frames # 返回初始状态、初始收益

    def step(self, action: int) -> Tuple[TensorObs, int, bool]: # step函数将动作action在环境中执行，并返回下一个状态、奖励和一个bool值（指示是否结束）
        action = action + 1 if not action == 0 else 0
        obs, reward, done, _ = self.__env.step(action) # 从已有的breakout模型得到下一状态、收益、是否结束
        return self.to_tensor(obs), reward, done

    def get_frame(self) -> GymImg: # 返回帧
        """get_frame renders the current game frame."""
        return Image.fromarray(self.__env.render(mode="rgb_array"))

    @staticmethod # 静态函数，下文同
    def to_tensor(obs: GymObs) -> TensorObs:
        """to_tensor converts an observation to a torch tensor."""
        return torch.from_numpy(obs).view(1, 84, 84)

    @staticmethod
    def get_action_dim() -> int: # 返回行为的维度：3
        """get_action_dim returns the reduced number of actions."""
        return 3

    @staticmethod
    def get_action_meanings() -> List[str]: # 返回行为的具体含义：静止、右移、左移
        """get_action_meanings returns the actual meanings of the reduced
        actions."""
        return ["NOOP", "RIGHT", "LEFT"]

    @staticmethod
    def get_eval_lives() -> int: # 返回每个玩家的测试次数（生命）：5
        """get_eval_lives returns the number of lives to consume in an
        evaluation round."""
        return 5

    @staticmethod
    def make_state(obs_queue: deque) -> TensorStack4: # 后4个状态拼接起来
        """make_state makes up a state given an obs queue."""
        return torch.cat(list(obs_queue)[1:]).unsqueeze(0)

    @staticmethod
    def make_folded_state(obs_queue: deque) -> TensorStack5: # 全部五个状态拼接起来
        """make_folded_state makes up an n_state given an obs queue."""
        return torch.cat(list(obs_queue)).unsqueeze(0)

    @staticmethod
    def show_video(path_to_mp4: str) -> None: # 生成可视化mp4文件
        """show_video creates an HTML element to display the given mp4 video in
        IPython."""
        mp4 = pathlib.Path(path_to_mp4)
        video_b64 = base64.b64encode(mp4.read_bytes())
        html = HTML_TEMPLATE.format(alt=mp4, data=video_b64.decode("ascii"))
        ipydisplay.display(ipydisplay.HTML(data=html))

    def evaluate(self, obs_queue: deque, agent: Agent, num_episode: int = 3, render: bool = False, ) -> Tuple[float, List[GymImg], ]: # 使用给定的代理运行游戏几次（3个玩家5个回合），并返回平均奖励和捕获的帧（实际上不返回帧）。
        """evaluate uses the given agent to run the game for a few episodes and
        returns the average reward and the captured frames."""
        self.__env = self.__env_eval
        ep_rewards = []
        frames = []
        for _ in range(self.get_eval_lives() * num_episode):
            observations, ep_reward, _frames = self.reset(render=render) # 初始化测试环境
            for obs in observations: obs_queue.append(obs)
            if render: frames.extend(_frames)
            done = False

            while not done: # 开始测试
                state = self.make_state(obs_queue).to(self.__device).float()
                action = agent.run(state) # 得到AI的下一个步骤
                obs, reward, done = self.step(action) # 得到下一状态、收益、是否结束

                ep_reward += reward
                obs_queue.append(obs)
                if render: frames.append(self.get_frame())

            ep_rewards.append(ep_reward) # 统计收益

        self.__env = self.__env_train
        return np.sum(ep_rewards) / num_episode, frames # 返回平均收益