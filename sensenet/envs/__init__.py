from sensenet.envs.registration import registry, register, make, spec
from sensenet.envs.handroid.hand_env import HandEnv

register(
    id='HandEnv-v0',
    entry_point='sensenet.envs.handroid:HandEnv',
    max_episode_steps=1000,
    reward_threshold=5000.0,
    )
