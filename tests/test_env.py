import sys
sys.path.append('..')
import gym
from env import SenseEnv
def test_environments():
    tenv = SenseEnv()
    env = gym.make("CartPole-v0")
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    tinput_dim = tenv.observation_space()
    toutput_dim = len(tenv.action_space())
    assert tinput_dim > 0
    print("gym observation space: ",input_dim)
    print("gym action space: ",output_dim)
    print("touch observation space: ",tinput_dim)
    print("touch action space: ",toutput_dim)
    state = env.reset()
    tstate = tenv.reset()
    print("gym state:",state)
    print("touch state:",tstate)
    #state, reward, done, _ = env.step(action[0,0])
