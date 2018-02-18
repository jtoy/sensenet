import sensenet
from sensenet.envs.handroid.hand_env import HandEnv
from sensenet import spaces
class CrazyEnv(HandEnv):
    def __init__(self,options={}):
        self.action_space = spaces.Discrete(1)
    def step(self,action):
        return [0],0,True,[42]
    def reset(self,opts={}):
        return [0]
