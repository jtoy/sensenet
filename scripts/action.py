import sys
sys.path.append('..')
import sensenet
if len(sys.argv) >= 2:
    env_id = sys.argv[1]
    print("loading env", env_id)
else:
    print("must pass env as arg")
env = sensenet.make(env_id,{'render':True})
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

