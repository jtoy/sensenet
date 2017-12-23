from sensenet.envs.registration import registry, register, make, spec
from sensenet.envs.handroid.hand_env import HandEnv
import os,inspect,sys,glob

register(
    id='HandEnv-v0',
    entry_point='sensenet.envs.handroid:HandEnv',
    max_episode_steps=1000,
    reward_threshold=5000.0,
    )
register(
    id='IndexFingerHandEnv-v0',
    entry_point='sensenet.envs.handroid:IndexFingerHandEnv',
    max_episode_steps=1000,
    reward_threshold=5000.0,
    )
register(
    id='BlankEnv-v0',
    entry_point='sensenet.envs.handroid:BlankEnv',
    )

# if folder envs from current code exists, register it
cwd = os.getcwd()
if os.path.isdir(cwd+"/envs"):
    files = glob.glob(cwd+"/envs/*.py")
    for f in files:
        #TODO read metadata
        id_name = sys.argv[0].split(".")[0]
        entry_point_name = ""
        for name in f.split("/")[-1].split(".")[0].split("_"):
            entry_point_name += name.capitalize()

        register(id=entry_point_name,entry_point=entry_point_name)
