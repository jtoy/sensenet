import sensenet
from sensenet import envs
def test_envs():
    print(envs.registry.all())
    env = sensenet.make("HandEnv-v0")
    env2 = sensenet.make("MyEnv")
