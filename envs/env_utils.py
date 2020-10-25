"""
Utils file for OpenAi gym envrionments.
"""


class WrapFrameSkip():
    """
    Wraps OpenAi gym environments to skip frames. This is also know as action repeat.
    """
    def __init__(self, env, frameskip):
        assert frameskip >= 1
        self._env = env
        self._frameskip = frameskip
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self):
        return self._env.reset()

    def step(self, action):
        sum_rew = 0
        for _ in range(self._frameskip):
            obs, rew, done, info = self._env.step(action)
            sum_rew += rew
            if done:
                break
        return obs, sum_rew, done, info

    def render(self, mode='human'):
        return self._env.render(mode=mode)

    def close(self):
        self._env.close()
