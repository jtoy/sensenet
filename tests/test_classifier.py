import sensenet
from sensenet import envs
def test_class_count():
    env = sensenet.make("HandEnv-v0",{'data_path':'data/1class_folder'})
    assert env.classification_n() == 1
    env = sensenet.make("HandEnv-v0",{'data_path':'data/2class_folder'})
    assert env.classification_n() == 2
