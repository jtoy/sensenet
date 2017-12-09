import sys
sys.path.append('..')

#from env import SenseEnv
#env = SenseEnv({'render':True,'debug':True,'obj_path':'../tests/data/pyramid.stl'})

import sensenet
from sensenet.envs.handroid.hand_env import HandEnv
env = HandEnv({'render':True})
#from sensenet.envs.handroid.index_finger_hand_env import IndexFingerHandEnv
#env = IndexFingerHandEnv({'render':True})

while (1):
  key = env.getKeyboardEvents()
  n = -1
  if len(key.keys()) >= 2:
    m = 0
    if 65307 in key.keys(): #shift
      m = 10
    elif 65306 in key.keys(): #control
      m = 20
    for k in key.keys():
      if k in range(48,58):
        n = k-48+m
  else:
    for k in key.keys():
      if k == 113: #q
        if action_mode == True:
          action_mode = False
        else:
          action_mode = True
      elif k in range(48,58):
        n = k-48
  #      print("new number",n)
      else:
        print("key: ",k)
  if n > -1:
   observation,reward,done,info = env.step(n)

