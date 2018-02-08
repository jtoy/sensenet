import sys
sys.path.append('..')
import sensenet,argparse

parser = argparse.ArgumentParser(description='SenseNet')
parser.add_argument('--render', action='store_false', help='render the environment')
parser.add_argument('--debug', action='store_true', help='debug')
parser.add_argument('--environment','-e', type=str, default="HandEnv-v0")
args = parser.parse_args()
env = sensenet.make(args.environment,{'render':args.render,'debug':args.debug})
env.reset()
while (1):
    key = env.getKeyboardEvents()
    n = -1
    if len(key.keys()) >= 2:
        m = 0
        if 65307 in key.keys(): #shift
            m = 10
        elif 65306 in key.keys(): #control
            m = 20
        for k in key.keys():
            if k in range(48,58):
              n = k-48+m
    else:
        for k in key.keys():
            if k == 113: #q
                if action_mode == True:
                    action_mode = False
                else:
                    action_mode = True
            elif k in range(48,58):
                n = k-48
            else:
                print("key: ",k)
    if n > -1:
        observation,reward,done,info = env.step(n)

