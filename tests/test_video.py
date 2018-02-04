import sys
sys.path.append('..')
import os
import sensenet
import random
from sensenet.envs.handroid.hand_env import HandEnv
def test_environments():
  #test if we can load an arbitrary path
  name = str(random.random())+".mp4"
  assert !os.path.exists(name)
  env = HandEnv({'video':name,'max_steps':10})
  for i in range(10):
      env.step()
  assert os.path.exists(name)
