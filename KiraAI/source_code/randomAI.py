import logging
import random

from pyftg.aiinterface.ai_interface import AIInterface
from pyftg.aiinterface.command_center import CommandCenter
from pyftg.models.audio_data import AudioData
from pyftg.models.frame_data import FrameData
from pyftg.models.game_data import GameData
from pyftg.models.key import Key
from pyftg.models.round_result import RoundResult
from pyftg.models.screen_data import ScreenData

logger = logging.getLogger(__name__)


class KickAI(AIInterface):
    def __init__(self):
        super().__init__()
        self.blind_flag = True
        # 标签与动作映射
        self.actions = {
            0: "AIR_A",
            1: "BACK_JUMP",
            2: "DASH",
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
            action = random.randint(1, 16)

            self.cc.command_call(self.actions[action])
    
    def round_end(self, round_result: RoundResult):
        logger.info(f"round end: {round_result}")

    def game_end(self):
        logger.info("game end")
