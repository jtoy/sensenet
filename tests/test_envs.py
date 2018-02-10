import sensenet
from sensenet import envs
def test_envs():
    print(envs.registry.all())
    env = sensenet.make("HandEnv-v0")
    env2 = sensenet.make("MyEnv")
    env3 = sensenet.make("CrazyEnv")
def test_can_list():
    envids = [spec.id for spec in envs.registry.all()]
    assert len(envids) > 0
