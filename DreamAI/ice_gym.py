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
from TestAI import TestAI

ICE_BAT = "..\\DF7beta\\run-windows-amd64.bat"

import numpy as np
import librosa

class MelSpecEncoder:
    def __init__(self, sampling_rate=48000):
        self.sampling_rate = sampling_rate
        
        self.window_size = int(self.sampling_rate * 0.025)  # 25 ms window
        self.hop_size = int(self.sampling_rate * 0.01)     # 10 ms hop
        self.n_fft = int(self.sampling_rate * 0.025)       # FFT size equal to window size
        self.n_mels = 80                                   # Number of Mel bands
        
        # Define Mel filter parameters
        self.f_min = 20                                    # Minimum frequency
        self.f_max = 7600                                  # Maximum frequency

    def numpy_to_mel(self, audio):
        """
        Converts a stereo numpy audio array to a 2-channel Mel spectrogram.
        Args:
            audio (numpy.ndarray): Input audio array with shape (n_samples, 2).

        Returns:
            numpy.ndarray: 2-channel Mel spectrogram with shape (2, n_mels, time_frames).
        """
        if audio.shape[1] != 2:
            raise ValueError("Input audio must have two channels (stereo).")

        # Process left and right channels separately
        left_channel = audio[:, 0]
        right_channel = audio[:, 1]

        # Compute Mel spectrogram for each channel
        mel_left = librosa.feature.melspectrogram(
            y=left_channel,
            sr=self.sampling_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max
        )
        mel_right = librosa.feature.melspectrogram(
            y=right_channel,
            sr=self.sampling_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max
        )
        mel_avg = (mel_left + mel_right) / 2

        # Convert to decibels
        mel_left_db = librosa.power_to_db(mel_left, ref=np.max)
        mel_right_db = librosa.power_to_db(mel_right, ref=np.max)
        mel_avg_db = librosa.power_to_db(mel_avg, ref=np.max)

        # Stack along the first dimension to create a 2-channel spectrogram
        mel_spectrogram_stereo = np.stack([mel_left_db, mel_avg_db, mel_right_db], axis=0)
        return mel_spectrogram_stereo

class ICEEnv(gym.Env):
    """
    Template for a custom Gym environment.
    """

    def __init__(self, sound_agent, p1_agent : WrappedAgent, frame_stack = 48, p2_agent = "MctsAi23i"):
        super(ICEEnv, self).__init__()
        self.ice_process = subprocess.Popen(ICE_BAT, shell=True)
        # Define the agents
        self.sound_agent = sound_agent
        self.mel_encoder = MelSpecEncoder()
        self.p1_agent = p1_agent
        self.frame_stack = frame_stack
        self.p1_name = "Dreamer"
        self.p2_agent = p2_agent
        self.gateway = Gateway()
        self.gateway.register_ai(self.p1_name, self.p1_agent)
        self.gateway.register_sound(self.sound_agent)
        self.resize_op = T.Resize((64, 64), interpolation=T.InterpolationMode.BILINEAR)

        # Define the action space (e.g., discrete actions)
        self.action_space = spaces.Discrete(len(p1_agent.actions))
                
        self.sound_task = asyncio.create_task(self.gateway.start_sound())
        self.gateway_task = asyncio.create_task(self.gateway.run_game(["ZEN", "ZEN"], [self.p1_name, self.p2_agent], 5))
    
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
    
    async def reset(self):
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
        self.ice_process = subprocess.Popen(ICE_BAT, shell=True)
        await self.gateway.close_game()

        # This is essential to ensure that the environment is properly reset!
        await asyncio.sleep(3)

        self.p1_agent.reset()
        self.gateway = Gateway()
        self.gateway.register_ai(self.p1_name, self.p1_agent)
        self.gateway.register_sound(self.sound_agent)
        self.gateway_task = asyncio.create_task(self.gateway.run_game(["ZEN", "ZEN"], [self.p1_name, self.p2_agent], 5))
        self.sound_task = asyncio.create_task(self.gateway.start_sound())
        while len(self.p1_agent.states) < self.frame_stack:
            await asyncio.sleep(0.05)
        # self.p1_agent.latest_state => nparr. (800, 2), cat to (800*40, 2)
        stacked_states = np.concatenate(self.p1_agent.states[-self.frame_stack:], axis=0)
        mel_spec = self.mel_encoder.numpy_to_mel(stacked_states)
        # reshape from (3, 80, 81) to (3, 80, 80)
        mel_spec = mel_spec[:, :, :-1]
        mel_spec -= mel_spec.min()
        mel_spec /= mel_spec.max()        
        # resize from 80x80 to 64x64
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

    async def step(self, action):
        """
        Execute an action and update the environment state.
        """
        self.p1_agent.latest_action = action
        current_state_size = len(self.p1_agent.states)
        while len(self.p1_agent.states) == current_state_size:
            await asyncio.sleep(0.01)
        reward = self.p1_agent.latest_reward
        while len(self.p1_agent.states) < self.frame_stack:
            await asyncio.sleep(0.05)
        mel_spec = self.mel_encoder.numpy_to_mel(np.concatenate(self.p1_agent.states[-self.frame_stack:], axis=0))
        mel_spec = mel_spec[:, :, :-1]
        mel_spec -= mel_spec.min()
        mel_spec /= mel_spec.max()
        mel_spec = torch.tensor(mel_spec)
        mel_spec = self.resize_op(mel_spec.unsqueeze(0)).squeeze(0).numpy()
        mel_spec *= 255
        mel_spec = mel_spec.astype(np.int8)
        mel_spec = mel_spec.transpose(1, 2, 0)
        obs = {
            "image": mel_spec,
            "is_first": False,
            "is_last": self.p1_agent.currentFrameNum >= 3570,
            "is_terminal": self.p1_agent.end_flag
        }
        return obs, reward, self.p1_agent.currentFrameNum >= 3570, {}
    
    def close(self):
        self.gateway_task.cancel()
        try:
            parent = psutil.Process(self.ice_process.pid)
            for child in parent.children(recursive=True):
                 child.terminate()
            parent.terminate()
        except psutil.NoSuchProcess:
            pass

# import random
# async def monitor(env, agent):
#     while not agent.end_flag:
#         act = random.randint(0, 10)
#         result = await env.step(act)
#         print(result[1])
#         await asyncio.sleep(0.1)
#         pass
        
# async def main():
#     agent = WrappedAgent()
#     sound_agent = SampleSoundGenAI()
#     env = ICEEnv(sound_agent, agent)
#     await asyncio.sleep(5)
#     await env.reset()
#     await asyncio.gather(env.gateway_task, asyncio.create_task(monitor(env, agent)))

# asyncio.run(main())
