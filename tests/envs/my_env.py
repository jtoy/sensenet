import sensenet
from sensenet import spaces
import numpy as np
class MyEnv(sensenet.SenseEnv):
    def __init__(self,options={}):
        self.action_space = spaces.Discrete(1)
    def step(self,action):
        return [0],0,True,[42]
    def _reset(self,opts={}):
        return [0]
