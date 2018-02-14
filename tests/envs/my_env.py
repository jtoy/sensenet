import sensenet
from sensenet import spaces
class MyEnv(sensenet.SenseEnv):
    def __init__(self,options={}):
        self.action_space = spaces.Discrete(1)
