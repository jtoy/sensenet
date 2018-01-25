import sys
sys.path.append('..')
import sensenet
from sensenet.envs.handroid.hand_env import HandEnv
def test_environments():
  #test if we can load an arbitrary path
  env = HandEnv({'data_path':'./data/good_folder'})
  assert env.class_label != None
def test_reset_can_set_obj():
  #test if we can load an arbitrary path
  env = sensenet.make("TouchWandEnv-v0",{'obj_path':'./data/cube.stl'})
  assert env.options['obj_path'], "./data/cube.stl"
  env._reset({'obj_path':'./data/sphere.stl'})
  assert env.options['obj_path'], "./data/sphere.stl"
