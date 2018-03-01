import sensenet
from sensenet import envs
def test_core_envs():
    env = sensenet.make("HandEnv-v0")
    assert env.action_space.n
    reset_observation = env.reset()
    assert(reset_observation is not None)
    observation = env.step(0)
    assert observation != None

def test_local_folder_envs():
    env = sensenet.make("MyEnv")
    assert env.action_space.n
    reset_observation = env.reset()
    assert(reset_observation is not None)
    observation = env.step(0)
    assert observation != None

    env = sensenet.make("CrazyEnv")
    assert env.action_space.n
    reset_observation = env.reset()
    assert(reset_observation is not None)
    observation = env.step(0)
    assert observation != None
def test_can_list():
    print(envs.registry.all())
    envids = [spec.id for spec in envs.registry.all()]
    assert len(envids) > 0
def test_can_list_local_envs():
    local_envs = [spec.id for spec in envs.registry.local_envs()]
    all_envs = [spec.id for spec in envs.registry.all()]
    assert len(local_envs) > 0
    assert len(all_envs) > len(local_envs)
