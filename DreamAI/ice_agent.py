import numpy as np
import torch
from pyftg.aiinterface.ai_interface import AIInterface
from pyftg.aiinterface.command_center import CommandCenter
from pyftg.models.audio_data import AudioData
from pyftg.models.frame_data import FrameData
from pyftg.models.key import Key
from pyftg.models.round_result import RoundResult
from pyftg.models.screen_data import ScreenData

from ppo_src.config import GATHER_DEVICE
from ppo_src.dataclasses.collect_data_helper import CollectDataHelper
from ppo_src.models.recurrent_actor import RecurrentActor
from ppo_src.models.recurrent_critic import RecurrentCritic
from loguru import logger
import time
import torchvision.transforms as T
from dreamer import Dreamer
from mel_encoder import MelSpecEncoder
from pyftg import (AIInterface, AudioData, CommandCenter, FrameData, GameData,
                   Key, RoundResult, ScreenData)

class RandomAgent(AIInterface):
    def __init__(self):
        super().__init__()
        self.blind_flag = True
        self.action_response_map = {
            "FORWARD_WALK":0.2,
            "DASH":0.2,
            "BACK_STEP":0.2,
            "CROUCH":0.2,
            "JUMP":0.8,
            "FOR_JUMP":0.8,
            "BACK_JUMP":0.8,
            "STAND_GUARD":0.5,
            "CROUCH_GUARD":0.5,
            "AIR_GUARD":0.2,
            "STAND_A":0.2,
            "STAND_B":0.2,
            "THROW_A":0.5,
            "THROW_B":0.5,
            "CROUCH_A":0.2,
            "CROUCH_B":0.2,
            "STAND_FA":0.5,
            "STAND_FB":0.5,
            "CROUCH_FA":0.5,
            "CROUCH_FB":0.5,
            "STAND_F_D_DFA":0.5,
            "STAND_F_D_DFB":0.5,
            "STAND_D_DB_BA":0.8,
            "STAND_D_DB_BB":0.5,
            "STAND_D_DF_FA":0.8,
            "STAND_D_DF_FB":0.8,
            "STAND_D_DF_FC":0.8,
            "AIR_A":0.2,
            "AIR_B":0.2,
            "AIR_DA":0.2,
            "AIR_DB":0.2,
            "AIR_FA":0.5,
            "AIR_FB":0.5,
            "AIR_UA":0.5,
            "AIR_UB":0.8,
            "AIR_F_D_DFA":0.8,
            "AIR_F_D_DFB":0.8,
            "AIR_D_DB_BA":0.8,
            "AIR_D_DB_BB":0.8,
            "AIR_D_DF_FA":0.8,
            "AIR_D_DF_FB":0.8,
        }
        self.actions = list(self.action_response_map.keys())

    def name(self) -> str:
        return self.__class__.__name__

    def is_blind(self) -> bool:
        return self.blind_flag

    def initialize(self, game_data: GameData, player_number: int):
        logger.info("initialize")
        self.cc = CommandCenter()
        self.key = Key()
        self.player = player_number

    def get_non_delay_frame_data(self, frame_data: FrameData):
        pass

    def input(self) -> Key:
        return self.key

    def get_information(self, frame_data: FrameData, is_control: bool):
        self.frame_data = frame_data
        self.cc.set_frame_data(self.frame_data, self.player)

    def get_screen_data(self, screen_data: ScreenData):
        self.screen_data = screen_data

    def get_audio_data(self, audio_data: AudioData):
        self.audio_data = audio_data

    def processing(self):
        if self.frame_data.empty_flag or self.frame_data.current_frame_number <= 0:
            return

        if self.cc.get_skill_flag():
            self.key = self.cc.get_skill_key()
        else:
            self.key.empty()
            self.cc.skill_cancel()
            random_action = np.random.randint(0, len(self.actions))
            self.cc.command_call(self.actions[random_action])
    
    def round_end(self, round_result: RoundResult):
        logger.info(f"round end: {round_result}")

    def game_end(self):
        logger.info("game end")
        
    def close(self):
        pass


class WrappedAgent(AIInterface):
    def __init__(self, **kwargs):
        self.device = GATHER_DEVICE
        self.trajectories_data = None

        self.action_response_map = {
            "FORWARD_WALK":0.2,
            "DASH":0.2,
            "BACK_STEP":0.2,
            "CROUCH":0.2,
            "JUMP":0.8,
            "FOR_JUMP":0.8,
            "BACK_JUMP":0.8,
            "STAND_GUARD":0.5,
            "CROUCH_GUARD":0.5,
            "AIR_GUARD":0.2,
            "STAND_A":0.2,
            "STAND_B":0.2,
            "THROW_A":0.5,
            "THROW_B":0.5,
            "CROUCH_A":0.2,
            "CROUCH_B":0.2,
            "STAND_FA":0.5,
            "STAND_FB":0.5,
            "CROUCH_FA":0.5,
            "CROUCH_FB":0.5,
            "STAND_F_D_DFA":0.5,
            "STAND_F_D_DFB":0.5,
            "STAND_D_DB_BA":0.8,
            "STAND_D_DB_BB":0.5,
            "STAND_D_DF_FA":0.8,
            "STAND_D_DF_FB":0.8,
            "STAND_D_DF_FC":0.8,
            "AIR_A":0.2,
            "AIR_B":0.2,
            "AIR_DA":0.2,
            "AIR_DB":0.2,
            "AIR_FA":0.5,
            "AIR_FB":0.5,
            "AIR_UA":0.5,
            "AIR_UB":0.8,
            "AIR_F_D_DFA":0.8,
            "AIR_F_D_DFB":0.8,
            "AIR_D_DB_BA":0.8,
            "AIR_D_DB_BB":0.8,
            "AIR_D_DF_FA":0.8,
            "AIR_D_DF_FB":0.8,
        }
        self.actions = list(self.action_response_map.keys())
        self.audio_data = None
        self.raw_audio_memory = None
        self.just_inited = True
        self.pre_framedata: FrameData = None
        self.nonDelay: FrameData = None

        self.latest_state = None
        self.states = []
        self.latest_reward = 0
        self.rewards = []
        self.latest_action = None
        self.end_flag = False

        self.round_count = 0
    
    def name(self) -> str:
        return self.__class__.__name__
    
    def is_blind(self) -> bool:
        return False

    def initialize(self, gameData, player):
        # Initializng the command center, the simulator and some other things
        self.inputKey = Key()
        self.frameData = FrameData()
        self.cc = CommandCenter()
        self.player = player  # p1 == True, p2 == False
        self.gameData = gameData
        self.isGameJustStarted = True

        self.latest_state = None
        self.states = []
        self.latest_reward = 0
        self.rewards = []
        self.latest_action = None
        self.end_flag = False

    def close(self):
        pass

    def get_non_delay_frame_data(self, non_delay: FrameData):
        self.pre_framedata = self.nonDelay if self.nonDelay is not None else non_delay
        self.nonDelay = non_delay

    def get_information(self, frame_data: FrameData, is_control: bool):
        # Load the frame data every time getInformation gets called
        self.frameData = frame_data
        self.cc.set_frame_data(self.frameData, self.player)
        # nonDelay = self.frameData
        self.isControl = is_control
        self.currentFrameNum = self.frameData.current_frame_number  # first frame is 14

    def round_end(self, round_result: RoundResult):
        logger.info(round_result.remaining_hps[0])
        logger.info(round_result.remaining_hps[1])
        logger.info(round_result.elapsed_frame)
        self.just_inited = True
        obs = self.raw_audio_memory
        # final step
        # state = torch.tensor(obs, dtype=torch.float32)
        terminal = 1
        true_reward = self.get_reward()
        self.raw_audio_memory = None
        self.round_count += 1 

        self.latest_state = None
        self.states = []
        self.latest_reward = 0
        self.latest_action = None
        self.end_flag = True
        logger.info('Finished {} round'.format(self.round_count))
    
    def game_end(self):
        self.end_flag = True

    def get_screen_data(self, screen_data: ScreenData):
        pass

    def input(self):
        return self.inputKey

    @torch.no_grad()
    def processing(self):
        # process audio
        try:
            np_array = np.frombuffer(self.audio_data.raw_data_bytes, dtype=np.float32)
            raw_audio = np_array.reshape((2, 1024))
            raw_audio = raw_audio.T
            raw_audio = raw_audio[:800, :]
        except Exception:
            raw_audio = np.zeros((800, 2))
        if self.raw_audio_memory is None:
            self.raw_audio_memory = raw_audio
        else:
            self.raw_audio_memory = np.vstack((raw_audio, self.raw_audio_memory))
            self.raw_audio_memory = self.raw_audio_memory[:800, :]

        # append so that audio memory has the first shape of n_frame
        increase = (800 - self.raw_audio_memory.shape[0]) // 800
        for _ in range(increase):
            self.raw_audio_memory = np.vstack((np.zeros((800, 2)), self.raw_audio_memory))

        obs = self.raw_audio_memory
        if self.just_inited:
            self.just_inited = False
            if obs is None:
                obs = np.zeros((800, 2))
            terminal = 1
            self.latest_state = obs.copy()
            self.states.append(self.latest_state)
        elif obs is None:
            obs = np.zeros((800, 2))
            self.latest_state = obs.copy()
            self.states.append(self.latest_state)
        else:
            terminal = 0
            reward = self.get_reward()
            self.latest_state = obs.copy()
            self.states.append(self.latest_state)
            self.latest_reward = reward
            self.rewards.append(reward)
            pass
        
        if self.cc.get_skill_flag():
            self.inputKey = self.cc.get_skill_key()
            return
        # put to helper      
        self.inputKey.empty()
        self.cc.skill_cancel()
        if self.latest_action is not None:
            self.cc.command_call(self.actions[self.latest_action])
            self.inputKey = self.cc.get_skill_key()

    def get_reward(self):
        offence_reward = self.pre_framedata.get_character(not self.player).hp - self.nonDelay.get_character(not self.player).hp
        defence_reward = self.nonDelay.get_character(self.player).hp - self.pre_framedata.get_character(self.player).hp
        return offence_reward + defence_reward

    def set_last_hp(self):
        self.last_my_hp = self.nonDelay.get_character(self.player).hp
        self.last_opp_hp = self.nonDelay.get_character(not self.player).hp

    def get_audio_data(self, audio_data: AudioData):
        self.audio_data = audio_data
    
    def reset(self):
        self.latest_state = None
        self.states = []
        self.latest_reward = 0
        self.rewards = []
        self.latest_action = None
        self.end_flag = True


class EntityAgent(AIInterface):
    def __init__(self, **kwargs):
        self.device = GATHER_DEVICE
        self.trajectories_data = None

        self.action_response_map = {
            "FORWARD_WALK":0.2,
            "DASH":0.2,
            "BACK_STEP":0.2,
            "CROUCH":0.2,
            "JUMP":0.8,
            "FOR_JUMP":0.8,
            "BACK_JUMP":0.8,
            "STAND_GUARD":0.5,
            "CROUCH_GUARD":0.5,
            "AIR_GUARD":0.2,
            "STAND_A":0.2,
            "STAND_B":0.2,
            "THROW_A":0.5,
            "THROW_B":0.5,
            "CROUCH_A":0.2,
            "CROUCH_B":0.2,
            "STAND_FA":0.5,
            "STAND_FB":0.5,
            "CROUCH_FA":0.5,
            "CROUCH_FB":0.5,
            "STAND_F_D_DFA":0.5,
            "STAND_F_D_DFB":0.5,
            "STAND_D_DB_BA":0.8,
            "STAND_D_DB_BB":0.5,
            "STAND_D_DF_FA":0.8,
            "STAND_D_DF_FB":0.8,
            "STAND_D_DF_FC":0.8,
            "AIR_A":0.2,
            "AIR_B":0.2,
            "AIR_DA":0.2,
            "AIR_DB":0.2,
            "AIR_FA":0.5,
            "AIR_FB":0.5,
            "AIR_UA":0.5,
            "AIR_UB":0.8,
            "AIR_F_D_DFA":0.8,
            "AIR_F_D_DFB":0.8,
            "AIR_D_DB_BA":0.8,
            "AIR_D_DB_BB":0.8,
            "AIR_D_DF_FA":0.8,
            "AIR_D_DF_FB":0.8,
        }
        self.actions = list(self.action_response_map.keys())
        self.audio_data = None
        self.raw_audio_memory = None
        self.just_inited = True
        self.pre_framedata: FrameData = None
        self.nonDelay: FrameData = None

        self.latest_state = None
        self.states = []
        self.latest_reward = 0
        self.rewards = []
        self.latest_action = None
        self.first_flag = True
        self.end_flag = False

        self.round_count = 0
        self.dreamer : Dreamer = None
        self.dreamer_state = None
        self.frame_stack = 120
        self.mel_encoder = MelSpecEncoder()
        self.mel_min, self.mel_max = -80, 0
        self.resize_op = T.Resize((64, 64), interpolation=T.InterpolationMode.BILINEAR)


    def set_dreamer(self, dreamer: Dreamer):
        # deep copy
        self.dreamer = dreamer
        self.dreamer.eval()
        self.dreamer.to(self.device)
    
    def dreamer_load(self, state_dict: dict):
        self.reset()
        self.dreamer.load_state_dict(state_dict)
        self.dreamer.eval()
        self.dreamer.to(self.device)
        
    
    def name(self) -> str:
        return self.__class__.__name__
    
    def is_blind(self) -> bool:
        return False

    def initialize(self, gameData, player):
        # Initializng the command center, the simulator and some other things
        self.inputKey = Key()
        self.frameData = FrameData()
        self.cc = CommandCenter()
        self.player = player  # p1 == True, p2 == False
        self.gameData = gameData
        self.isGameJustStarted = True

        self.latest_state = None
        self.states = []
        self.latest_reward = 0
        self.rewards = []
        self.latest_action = None
        self.end_flag = False
        self.first_flag = True
        self.dreamer_state = None

    def close(self):
        pass

    def get_non_delay_frame_data(self, non_delay: FrameData):
        self.pre_framedata = self.nonDelay if self.nonDelay is not None else non_delay
        self.nonDelay = non_delay

    def get_information(self, frame_data: FrameData, is_control: bool):
        # Load the frame data every time getInformation gets called
        self.frameData = frame_data
        self.cc.set_frame_data(self.frameData, self.player)
        # nonDelay = self.frameData
        self.isControl = is_control
        self.currentFrameNum = self.frameData.current_frame_number  # first frame is 14

    def round_end(self, round_result: RoundResult):
        logger.info(round_result.remaining_hps[0])
        logger.info(round_result.remaining_hps[1])
        logger.info(round_result.elapsed_frame)
        self.just_inited = True
        obs = self.raw_audio_memory
        # final step
        # state = torch.tensor(obs, dtype=torch.float32)
        terminal = 1
        true_reward = self.get_reward()
        self.raw_audio_memory = None
        self.round_count += 1 

        self.latest_state = None
        self.states = []
        self.latest_reward = 0
        self.latest_action = None
        self.end_flag = True
        self.dreamer_state = None
        logger.info('Finished {} round'.format(self.round_count))
    
    def game_end(self):
        self.end_flag = True

    def get_screen_data(self, screen_data: ScreenData):
        pass

    def input(self):
        return self.inputKey

    @torch.no_grad()
    def processing(self):
        # process audio
        try:
            np_array = np.frombuffer(self.audio_data.raw_data_bytes, dtype=np.float32)
            raw_audio = np_array.reshape((2, 1024))
            raw_audio = raw_audio.T
            raw_audio = raw_audio[:800, :]
        except Exception:
            raw_audio = np.zeros((800, 2))
        if self.raw_audio_memory is None:
            self.raw_audio_memory = raw_audio
        else:
            self.raw_audio_memory = np.vstack((raw_audio, self.raw_audio_memory))
            self.raw_audio_memory = self.raw_audio_memory[:800, :]

        # append so that audio memory has the first shape of n_frame
        increase = (800 - self.raw_audio_memory.shape[0]) // 800
        for _ in range(increase):
            self.raw_audio_memory = np.vstack((np.zeros((800, 2)), self.raw_audio_memory))

        obs = self.raw_audio_memory
        if self.just_inited:
            self.just_inited = False
            if obs is None:
                obs = np.zeros((800, 2))
            terminal = 1
            self.latest_state = obs.copy()
            self.states.append(self.latest_state)
        elif obs is None:
            obs = np.zeros((800, 2))
            self.latest_state = obs.copy()
            self.states.append(self.latest_state)
        else:
            terminal = 0
            reward = self.get_reward()
            self.latest_state = obs.copy()
            self.states.append(self.latest_state)
            self.latest_reward = reward
            self.rewards.append(reward)
            pass
        if len(self.states) < self.frame_stack:
            pad_len = self.frame_stack - len(self.states)
            pad_data_single = np.zeros_like(self.states[-1])
            padded_data = np.concatenate([pad_data_single] * pad_len, axis=0)
            stacked_states = np.concatenate([padded_data, *self.states], axis=0)
        else:
            stacked_states = np.concatenate(self.states[-self.frame_stack:], axis=0)
        if self.cc.get_skill_flag():
            self.inputKey = self.cc.get_skill_key()
            return
        # put to helper      
        self.inputKey.empty()
        self.cc.skill_cancel()
        mel_spec = self.mel_encoder.numpy_to_mel(stacked_states)
        mel_spec = mel_spec[:, :, :-1]
        mel_spec = np.clip(mel_spec, self.mel_min, self.mel_max)
        mel_spec = (mel_spec - self.mel_min) / (self.mel_max - self.mel_min)
        mel_spec = torch.tensor(mel_spec)
        mel_spec = self.resize_op(mel_spec.unsqueeze(0)).squeeze(0).numpy()
        mel_spec *= 255
        mel_spec = mel_spec.astype(np.int8)
        mel_spec = mel_spec.transpose(1, 2, 0)
        obs = {
            "image": np.expand_dims(mel_spec, axis=0),
            "is_first": np.array([self.first_flag]),
            "is_last": np.array([False]),
            "is_terminal": np.array([False])
        }
        with torch.no_grad():
            action, self.dreamer_state = self.dreamer(obs, np.array([False]), self.dreamer_state, training=False)
        action = torch.squeeze(action['action']).cpu().numpy()
        action = np.argmax(action).item()
        self.first_flag = False
        self.latest_action = action
        self.cc.command_call(self.actions[self.latest_action])
        self.inputKey = self.cc.get_skill_key()

    def get_reward(self):
        offence_reward = self.pre_framedata.get_character(not self.player).hp - self.nonDelay.get_character(not self.player).hp
        defence_reward = self.nonDelay.get_character(self.player).hp - self.pre_framedata.get_character(self.player).hp
        return 1.1 * offence_reward + 0.9 * defence_reward # + time_reward

    def set_last_hp(self):
        self.last_my_hp = self.nonDelay.get_character(self.player).hp
        self.last_opp_hp = self.nonDelay.get_character(not self.player).hp

    def get_audio_data(self, audio_data: AudioData):
        self.audio_data = audio_data
    
    def reset(self):
        self.latest_state = None
        self.states = []
        self.latest_reward = 0
        self.rewards = []
        self.latest_action = None
        self.end_flag = True
        self.first_flag = True
        self.dreamer_state = None