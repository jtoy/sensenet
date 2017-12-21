import sys
sys.path.append('..')
import sensenet
from sensenet.envs.handroid.hand_env import HandEnv
def test_environments():
  #test if we can load an arbitrary path
  env = HandEnv({'data_path':'./data/good_folder'})
  assert env.class_label != None
