import numpy as np
import logging
import librosa
import tensorflow as tf
import os
from tensorflow.keras.models import load_model
from pyftg.aiinterface.ai_interface import AIInterface
from pyftg.aiinterface.command_center import CommandCenter
from pyftg.models.audio_data import AudioData
from pyftg.models.frame_data import FrameData
from pyftg.models.game_data import GameData
from pyftg.models.key import Key
from pyftg.models.round_result import RoundResult
from pyftg.models.screen_data import ScreenData

logger = logging.getLogger(__name__)
path = os.path.join('KirariAI', 'KiraAI', 'trained_model', 'my_model.keras')
# 将音频字节数据转换为浮点数数组
def bytes_to_audio(audio_sample_bytes):
    audio_sample = np.frombuffer(audio_sample_bytes, dtype=np.float32)
    return audio_sample

# 提取音频特征，确保形状为 (4, 13, 1)
def extract_features(audio_data, sample_rate):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13, n_fft=512, n_mels=40, fmax=8000)
    mfccs = np.pad(mfccs, ((0, max(0, 4 - mfccs.shape[0])), (0, max(0, 13 - mfccs.shape[1]))), mode='constant', constant_values=0)
    mfccs = mfccs[:4, :13]  # 截断或填充以确保形状为 (4, 13)
    mfccs = mfccs[..., np.newaxis]  # 添加新轴以确保形状为 (4, 13, 1)
    return mfccs

class testAI(AIInterface):
    def __init__(self):
        super().__init__()
        self.blind_flag = True
        self.model = load_model(path)
        #self.model = load_model('KirariAI/KiraAI/trained_model/my_model.keras')  # 加载自定义模型
        self.cc = CommandCenter()
        self.key = Key()
        self.player = None
        self.frame_data = None
        self.audio_data = None  # 用于存储音频数据
        self.action_count = 0  # 初始化动作计数器

        # 标签与动作映射
        self.actions = {
            0: "AIR_A",
            1: "DASH",
            2: "BACK_JUMP",
            3: "JUMP",
            4: "CROUCH",
            5: "THROW_A",
            6: "THROW_B",
            7: "STAND_A",
            8: "STAND_B",
            9: "STAND_FA",
            10: "STAND_FB",
            11: "STAND_GUARD",
            12: "AIR_GUARD",
            13: "THROW_HIT",
            14: "THROW_SUFFER",
            15: "STAND_A",
            16: "STAND_D_DF_FA"  # 默认动作
        }

    def name(self) -> str:
        return self.__class__.__name__

    def is_blind(self) -> bool:
        return self.blind_flag

    def initialize(self, game_data: GameData, player_number: int):
        logger.info("initialize")
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
        self.audio_data = audio_data  # 存储音频数据以供后续处理

    def handle_audio_data(self):
        try:
            audio_sample_bytes = self.audio_data.raw_data_bytes  # 直接使用 bytes 对象
            audio_data_np = bytes_to_audio(audio_sample_bytes)
            sample_rate = 44100  # 假设采样率为44100
            features = extract_features(audio_data_np, sample_rate)
            features = features[np.newaxis, ...]  # 添加批次维度
            prediction = self.model.predict(features)
            action_probabilities = prediction[0]

            # 确保 action_probabilities 和 self.actions 的长度一致
            if len(action_probabilities) > len(self.actions):
                action_probabilities = action_probabilities[:len(self.actions)]
            elif len(action_probabilities) < len(self.actions):
                action_probabilities = np.pad(action_probabilities, (0, len(self.actions) - len(action_probabilities)), mode='constant')

            # 引入随机性
            action = np.random.choice(len(self.actions), p=action_probabilities)

            return action

        except Exception as e:
            logger.error(f"Error in handle_audio_data: {e}")
            return 16  # 返回默认动作

    def processing(self):
        if self.frame_data.empty_flag or self.frame_data.current_frame_number <= 0:
            return

        if self.cc.get_skill_flag():
            self.key = self.cc.get_skill_key()
        else:
            self.key.empty()
            self.cc.skill_cancel()

            if self.audio_data is not None:
                self.action_count += 1  # 增加计数器
                
                # 添加更多随机性逻辑
                if self.action_count % 5 == 0:
                    action = np.random.choice(list(self.actions.keys()))  # 每五次随机选择一个动作
                elif self.action_count % 3 == 0:
                    action = 8  # 每三次强制执行 STAND_B
                else:
                    action = self.handle_audio_data()  # 处理音频数据并获取动作

                if action in self.actions:
                    print(self.actions[action])
                    self.cc.command_call(self.actions[action])
                else:
                    self.cc.command_call("STAND_B")

    def round_end(self, round_result: RoundResult):
        logger.info(f"round end: {round_result}")

    def game_end(self):
        logger.info("game end")
