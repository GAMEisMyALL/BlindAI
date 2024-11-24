import numpy as np
import tensorflow as tf
from keras._tf_keras.keras.models import load_model
#from tensorflow.keras.models import load_model
from pyftg.aiinterface.ai_interface import AIInterface
from pyftg.aiinterface.command_center import CommandCenter
from pyftg.models.audio_data import AudioData
from pyftg.models.frame_data import FrameData
from pyftg.models.game_data import GameData
from pyftg.models.key import Key
from pyftg.models.round_result import RoundResult
from pyftg.models.screen_data import ScreenData
import os
import librosa
import logging

logger = logging.getLogger(__name__)
path = os.path.join('KirariAI', 'KiraAI', 'trained_model', 'DQN_model.keras')
# Converts audio byte data to floating-point arrays
def bytes_to_audio(audio_sample_bytes):
    audio_sample = np.frombuffer(audio_sample_bytes, dtype=np.float32)
    return audio_sample

# Extract audio features, make sure the shape is (13,)
def extract_features(audio_data, sample_rate):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13, n_fft=512, n_mels=40, fmax=8000)
    mfccs = np.mean(mfccs.T, axis=0)
    return mfccs

class KiraAI(AIInterface):
    def __init__(self):
        super().__init__()
        self.model = load_model(path)  # Load the DQN model
        #self.model = load_model('KirariAI/KiraAI/trained_model/DQN_model.keras')  
        self.cc = CommandCenter()
        self.key = Key()
        self.player = None
        self.frame_data = FrameData.get_default_instance()
        self.audio_data = None
        self.epsilon = 0.2  
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
            11: "FOR_JUMP",
            12: "STAND_GUARD",
            13: "THROW_HIT",
            14: "THROW_SUFFER",
            15: "STAND_B",
            16: "STAND_D_DF_FA"  # default action
        }

    def name(self) -> str:
        return self.__class__.__name__

    def is_blind(self) -> bool:
        return True

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
        pass
        #self.screen_data = screen_data

    def get_audio_data(self, audio_data: AudioData):
        self.audio_data = audio_data

    def handle_audio_data(self):
        try:
            audio_sample_bytes = self.audio_data.raw_data_bytes
            audio_data_np = bytes_to_audio(audio_sample_bytes)
            sample_rate = 48000
            features = extract_features(audio_data_np, sample_rate)
            features = np.expand_dims(features, axis=0)  # Add a batch dimension

            if np.random.rand() < self.epsilon:
                action = np.random.choice(len(self.actions))  # randomly choose
            else:
                q_values = self.model.predict(features)
                action = np.argmax(q_values[0])
               # print("KiraAI:",action)

            return action

        except Exception as e:
            logger.error(f"Error in handle_audio_data: {e}")
            return 16  # default

    def processing(self):
        #if self.frame_data.empty_flag or self.frame_data.current_frame_number <= 0:
            #return

        if self.cc.get_skill_flag():
            self.key = self.cc.get_skill_key()
        else:
            self.key.empty()
            self.cc.skill_cancel()

            if self.audio_data is not None:
                action = self.handle_audio_data()

                if action in self.actions:
                    print(self.actions[action])
                    self.cc.command_call(self.actions[action])
                else:
                    self.cc.command_call("STAND_A")

    def round_end(self, round_result: RoundResult):
        logger.info(f"round end: {round_result}")

    def game_end(self):
        logger.info("game end")
