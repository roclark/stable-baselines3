import gym_super_mario_bros
from gym import Wrapper
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import A2C
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3.common.cmd_util import make_vec_env


class CustomReward(Wrapper):
    def __init__(self, env):
        super(CustomReward, self).__init__(env)
        self._current_score = 0

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        reward += (info['score'] - self._current_score) / 40.0
        self._current_score = info['score']
        if done:
            if info['flag_get']:
                print('We got it!!!!!')
                reward += 350.0
            else:
                reward -= 50.0
        return state, reward / 10.0, done, info

    def reset(self):
        """Reset the environment and return the initial observation."""
        return self.env.reset()


def mario_wrapper(env):
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = AtariWrapper(env, terminal_on_life_loss=False, clip_reward=False)
    env = CustomReward(env)
    return env


# env = make_vec_env('SuperMarioBros-1-4-v0', n_envs=16, seed=3994448089, start_index=0, monitor_dir=None, wrapper_class=mario_wrapper, env_kwargs=None, vec_env_cls=None, vec_env_kwargs=None)
# env = make_vec_env('SuperMarioBros-1-4-v0', n_envs=16, wrapper_class=mario_wrapper)
#env = gym_super_mario_bros.make('SuperMarioBros-1-4-v0')
#env = CustomReward(env)
env = make_vec_env('SuperMarioBros-1-4-v0', n_envs=16, seed=3994448089, wrapper_class=mario_wrapper)
env = VecFrameStack(env, n_stack=4)
env = VecTransposeImage(env)

model = A2C('CnnPolicy', env, verbose=1, vf_coef=0.25, ent_coef=0.01, policy_kwargs={'optimizer_class': RMSpropTFLike}, tensorboard_log='./mario')
model.learn(total_timesteps=20000000)
