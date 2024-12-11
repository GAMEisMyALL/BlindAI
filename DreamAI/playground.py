import asyncio
from asyncio import run

class SampleEnv:
    def __init__(self):
        self.states = []
        self.latest_action = None
        self.latest_reward = None

    async def reset(self):
        await asyncio.sleep(1)
        print("reset")

    async def step(self, action):
        await asyncio.sleep(2)
        print("step")

class AsyncDamy:
    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    async def step(self, action):
        return lambda: self._env.step(action)

    async def reset(self):
        return lambda: self._env.reset()
    
async def main():
    env = SampleEnv()
    damy = AsyncDamy(env)
    await asyncio.sleep(5)
    await damy.reset()
    await asyncio.gather(damy.step(0), damy.step(1))

asyncio.run(main())