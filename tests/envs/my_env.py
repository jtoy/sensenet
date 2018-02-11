import sensenet
class MyEnv(sensenet.SenseEnv):
    def __init__(self,options={}):
        self.action_space = 1
