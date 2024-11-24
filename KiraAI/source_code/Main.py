import asyncio
import typer
from typing_extensions import Annotated, Optional

from testAI import testAI
from randomAI import KickAI
from KiraAI import KiraAI
import os
from pyftg.utils.gateway import get_async_gateway
from pyftg.utils.logging import DEBUG, set_logging

from pathlib import Path
from dotenv import load_dotenv
from pyftg.socket.asyncio.generative_sound_gateway import GenerativeSoundGateway
from src.core import SampleSoundGenAI
from src.utils import setup_logging

app = typer.Typer(pretty_exceptions_enable=False)


async def start_game_process(host: str, port: int, use_grpc: bool, character: str = "ZEN", game_num: int = 1):
    gateway = get_async_gateway(host, port, use_grpc)
    agent1 = KiraAI()
    agent2 = testAI()
    gateway.register_ai("KiraAI", agent1)
    gateway.register_ai("testAI", agent2)
    await gateway.run_game([character, character], ["KiraAI", "testAI"], game_num)
    await gateway.close()

async def start_audio_process(port: int):
    gateway = GenerativeSoundGateway(port=port)
    sound_genai = SampleSoundGenAI()
    gateway.register(sound_genai)
    await gateway.run()
    await gateway.close()

async def main_process(host: str, game_port: int, audio_port: int, use_grpc: bool, character: str, game_num: int):
    await asyncio.gather(
        start_game_process(host, game_port, use_grpc, character, game_num),
        start_audio_process(audio_port)
    )

@app.command()
def main(
        host: Annotated[Optional[str], typer.Option(help="Host used by DareFightingICE")] = "127.0.0.1",
        game_port: Annotated[Optional[int], typer.Option(help="Port used by DareFightingICE for the game")] = 50051,
        audio_port: Annotated[Optional[int], typer.Option(help="Port used for the audio service")] = 12345,
        use_grpc: Annotated[Optional[bool], typer.Option(help="Use gRPC instead of socket")] = True,
        character: Annotated[Optional[str], typer.Option(help="Character to be used in the game")] = "ZEN",
        game_num: Annotated[Optional[int], typer.Option(help="Number of games to run")] = 1):
    asyncio.run(main_process(host, game_port, audio_port, use_grpc, character, game_num))

if __name__ == '__main__':
    set_logging(log_level=DEBUG)
    
   # Path(os.path.join('KirariAI', 'KiraAI', 'logs')).mkdir(exist_ok=True)
    load_dotenv()
    setup_logging()
    app()
