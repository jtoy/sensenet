#load a stl/obj file as argv[1]
import sys
sys.path.append('..')
import sensenet
from random import randint
env = sensenet.make("BlankEnv-v0",{'render':True,'obj_path':sys.argv[1]})
done = False
while (1):
    if done:
        env.reset()
    action = randint(0, env.action_space_n())
    observation,reward,done,info = env.step(action)
