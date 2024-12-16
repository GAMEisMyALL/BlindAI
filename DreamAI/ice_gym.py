import gym
from gym import spaces
import numpy as np
import psutil
from ice_agent import WrappedAgent
from pyftg.socket.aio.gateway import Gateway
import subprocess
import asyncio
import time
from sai_src.core import SampleSoundGenAI
import librosa
import torch
import torchvision.transforms as T
from ice_agent import EntityAgent, RandomAgent
import copy

ICE_BAT = "..\\DF7beta\\run-windows-amd64.bat"

import numpy as np
import librosa

from mel_encoder import MelSpecEncoder
class ICEEnv(gym.Env):
    """
    Template for a custom Gym environment.
    """

    def __init__(self, sound_agent, p1_agent : WrappedAgent, frame_stack = 120):
        super(ICEEnv, self).__init__()
        self.ice_process = subprocess.Popen(ICE_BAT, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Define the agents
        self.sound_agent = sound_agent
        self.mel_encoder = MelSpecEncoder()
        self.p1_agent = p1_agent
        self.frame_stack = frame_stack
        self.p1_name = "Dreamer"
        self.mcts_p2 = "MctsAi23i"
        self.random_p2 = RandomAgent()
        self.entity_p2 = EntityAgent()
        self.self_play_prob = 0.25
        self.entity_dreamer_set = False
        self.dreamer_state_dicts = []
        self.maximum_dict_len = 8
        self.gateway = Gateway()
        self.gateway.register_ai(self.p1_name, self.p1_agent)
        self.gateway.register_ai("RandAgent", self.random_p2)
        self.gateway.register_ai("EntityAgent", self.entity_p2)
        self.gateway.register_sound(self.sound_agent)
        self.resize_op = T.Resize((64, 64), interpolation=T.InterpolationMode.BILINEAR)
        self.mel_min = -80
        self.mel_max = 0

        # Define the action space (e.g., discrete actions)
        self.action_space = spaces.Discrete(len(p1_agent.actions))
                
        self.sound_task = asyncio.create_task(self.gateway.start_sound())
        self.gateway_task = asyncio.create_task(self.gateway.run_game(["ZEN", "ZEN"], [self.p1_name, self.mcts_p2], 5))
    
    def set_entity_dreamer(self, dreamer):
        self.entity_dreamer_set = True
        self.entity_p2.set_dreamer(dreamer)
    
    def add_dreamer_state_dict(self, state_dict):
        if len(self.dreamer_state_dicts) < self.maximum_dict_len:
            self.dreamer_state_dicts.append(copy.deepcopy(state_dict))
        else:
            self.dreamer_state_dicts.pop(0)
            self.dreamer_state_dicts.append(copy.deepcopy(state_dict))

    @property
    def observation_space(self):
        spaces = {
            "image": gym.spaces.Box(
                0, 255, (64, 64, 3), dtype=np.uint8
            ),
            "is_first": gym.spaces.Box(-2147483647, 2147483647, (1,), dtype=np.uint8),
            "is_last": gym.spaces.Box(-2147483647, 2147483647, (1,), dtype=np.uint8),
            "is_terminal": gym.spaces.Box(-2147483647, 2147483647, (1,), dtype=np.uint8),
        }
        return gym.spaces.Dict(spaces)
    
    async def reset(self, self_play_dreamer = None):
        """
        Reset the environment to its initial state.
        """
        self.gateway_task.cancel()
        self.sound_task.cancel()
        try:
            parent = psutil.Process(self.ice_process.pid)
            for child in parent.children(recursive=True):
                 child.terminate()
            parent.terminate()
        except psutil.NoSuchProcess:
            pass
        self.ice_process = subprocess.Popen(ICE_BAT, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        await self.gateway.close_game()

        # This is essential to ensure that the environment is properly reset!
        await asyncio.sleep(3)

        self.p1_agent.reset()
        self.gateway = Gateway()
        self.gateway.register_ai(self.p1_name, self.p1_agent)
        self.gateway.register_ai(self.p1_name, self.p1_agent)
        self.gateway.register_ai("RandAgent", self.random_p2)
        self.gateway.register_ai("EntityAgent", self.entity_p2)

        if self.entity_dreamer_set:
            if np.random.rand() < self.self_play_prob and type(self_play_dreamer) == type(self.entity_p2.dreamer):
                self.entity_p2.dreamer_load(copy.deepcopy(self_play_dreamer.state_dict()))
                opponent_name = "EntityAgent"
            else:
                state_dict_cnt = len(self.dreamer_state_dicts)
                choise_num = np.random.randint(0, state_dict_cnt + 2)
                if choise_num == 0:
                    opponent_name = "MctsAi23i"
                elif choise_num == 1:
                    opponent_name = "RandAgent"
                else:
                    selected_state_dict = self.dreamer_state_dicts[choise_num - 2]
                    self.entity_p2.dreamer_load(selected_state_dict)
                    opponent_name = "EntityAgent"
        else:
            opponent_name = "RandAgent" if np.random.rand() < 0.5 else "MctsAi23i"
        print(f"Opponent: {opponent_name}")
        self.gateway.register_sound(self.sound_agent)
        self.gateway_task = asyncio.create_task(self.gateway.run_game(["ZEN", "ZEN"], [self.p1_name, opponent_name], 5))
        self.sound_task = asyncio.create_task(self.gateway.start_sound())
        while len(self.p1_agent.states) == 0:
            await asyncio.sleep(0.05)
        if len(self.p1_agent.states) < self.frame_stack:
            pad_len = self.frame_stack - len(self.p1_agent.states)
            pad_data_single = np.zeros_like(self.p1_agent.states[-1])
            padded_data = np.concatenate([pad_data_single] * pad_len, axis=0)
            stacked_states = np.concatenate([padded_data, *self.p1_agent.states], axis=0)
        else:
            stacked_states = np.concatenate(self.p1_agent.states[-self.frame_stack:], axis=0)
        mel_spec = self.mel_encoder.numpy_to_mel(stacked_states)
        mel_spec = np.clip(mel_spec, self.mel_min, self.mel_max)
        mel_spec = (mel_spec - self.mel_min) / (self.mel_max - self.mel_min)
        # reshape from (3, 80, X) to (X, )
        mel_spec = torch.tensor(mel_spec)
        mel_spec = self.resize_op(mel_spec.unsqueeze(0)).squeeze(0).numpy()
        mel_spec = mel_spec.transpose(1, 2, 0)
        mel_spec *= 255
        mel_spec = mel_spec.astype(np.int8)

        obs = {
            "image": mel_spec,
            "is_first": True,
            "is_last": False,
            "is_terminal": False
        }
        return obs

    async def step(self, action, agent):
        """
        Execute an action and update the environment state.
        """
        self.p1_agent.latest_action = action
        current_reward_len = len(self.p1_agent.rewards)
        # 1.1s -> 60 frames, 0.1s -> (0.1/1.1)*60 = 5.45 frames -> round up to 6
        action_time = self.p1_agent.action_response_map[self.p1_agent.actions[action]] 
        frame_required = int(np.ceil(action_time / 1.1 * 60))
        # Very Important: Release the Fxcking Lock and Enable the Environment to Update Our Agent
        await asyncio.sleep(0.1)
        while len(self.p1_agent.rewards) - current_reward_len < frame_required \
            and self.p1_agent.currentFrameNum < 3420 \
            and not self.p1_agent.end_flag:
            await asyncio.sleep(0.1)
        reward = np.sum(self.p1_agent.rewards[current_reward_len:])
        while len(self.p1_agent.states) == 0:
            await asyncio.sleep(0.05)
        if len(self.p1_agent.states) < self.frame_stack:
            pad_len = self.frame_stack - len(self.p1_agent.states)
            pad_data_single = np.zeros_like(self.p1_agent.states[-1])
            padded_data = np.concatenate([pad_data_single] * pad_len, axis=0)
            stacked_states = np.concatenate([padded_data, *self.p1_agent.states], axis=0)
        else:
            stacked_states = np.concatenate(self.p1_agent.states[-self.frame_stack:], axis=0)
        mel_spec = self.mel_encoder.numpy_to_mel(stacked_states)
        mel_spec = np.clip(mel_spec, self.mel_min, self.mel_max)
        mel_spec = (mel_spec - self.mel_min) / (self.mel_max - self.mel_min)
        # reshape from (3, 80, X) to (X, )
        mel_spec = torch.tensor(mel_spec)
        mel_spec = self.resize_op(mel_spec.unsqueeze(0)).squeeze(0).numpy()
        mel_spec = mel_spec.transpose(1, 2, 0)
        mel_spec *= 255
        mel_spec = mel_spec.astype(np.int8)
        done = self.p1_agent.currentFrameNum >= 3420 or self.p1_agent.end_flag
        obs = {
            "image": mel_spec,
            "is_first": False,
            "is_last": done,
            "is_terminal": self.p1_agent.end_flag
        }
        return obs, reward, done, {}
    
    def close(self):
        self.gateway_task.cancel()
        try:
            parent = psutil.Process(self.ice_process.pid)
            for child in parent.children(recursive=True):
                 child.terminate()
            parent.terminate()
        except psutil.NoSuchProcess:
            pass
