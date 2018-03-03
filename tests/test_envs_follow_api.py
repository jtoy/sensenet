import sys
sys.path.append('..')
import sensenet
from sensenet import envs
from contextlib import contextmanager

def test_envs_follow_api():
    env_names = [spec.id for spec in envs.registry.all()]
    for name in env_names:
        print("loading env",name)
        env = sensenet.make(name)
        assert env.action_space.n

        reset_observation = env.reset()

        #TODO write  test to confirm observation space is the same for the reset and step function
        #assert env.observation_space == reset_observation.shape
   

        assert(reset_observation is not None)
        observation = env.step(0)
        assert observation != None

        #put any methods in here to test exceptions
        try:
            #end.seed(42)
            env.reset()
            env.render()
            assert True
        except:
            raise Exception('spam', 'eggs')

