import sys
sys.path.append('..')
import sensenet
import random
import pybullet as p

if len(sys.argv) > 1:
  envid = sys.argv[1]
else:
  envid = "HandEnv-v0"
if len(sys.argv) > 2:
  obj_path = sys.argv[2]
else:
  obj_path = None
#TODO add support for folders,classes, and saving data
env = sensenet.make(envid,{'render':True,'obj_path':obj_path})
env.reset()

forward = 3
left = 0
right = 1
up = 4
down = 5
indexup = 6
indexdown = 7
action_choices = [forward,left,right,up,down] #dont move index for now

training_sets = []
observations = []
actions = []
touch_count = 0
step = 0
episode = 0
plan_step = 0
winners = 0
plans = [([left]*5),([right] * 5),([up] * 5),([down] * 5),([indexup] * 5),([indexdown] * 6),] + (["random"] * 6)
def random_plan():
  a = []
  for i in range(5):
    a.append(random.randint(0,7))
  return a      
plan = None
while (1):
   al, _ = p.getBasePositionAndOrientation(env.agent)
   ol, _ = p.getBasePositionAndOrientation(env.obj_to_classify)
   xd = (al[0]-ol[0])/2
   yd = (al[1]-ol[1])/2
   zd = (al[2]-ol[2])/2
   #print("xd",xd,"yd",yd,"zd",zd)

   if env.is_touching() and plan_step == 0:
     plan = random.choice(plans)
     if plan == "random":
       plan = random_plan()
     action = plan[0]
     plan_step +=1
   elif plan_step >0:
     if len(plan) >= plan_step:
       #reset to a new plan
       plan_step = 0
       plan = random.choice(plans)
       if plan == "random":
         plan = random_plan()
     action = plan[plan_step]
     plan_step += 1
   elif not env.is_touching():
     if random.random() > 0.6 and zd >= xd:
     #elif random.random() > 0.3 and zd >= xd and zd >= yd:
       #print('z')
       action = down
     elif random.random() > 0.4 and xd >= yd:
     #if random.random() > 0.3 and xd >= yd and xd >= zd:
       #print('x')
       #action = forward
       action = left
     elif random.random() > 0.3 and yd >= xd:
     #elif random.random() > 0.3 and yd >= xd and yd >= zd:
       #print('y')
       #action = left
       action = forward
     else:
       #action = forward #should never get here
       action = random.choice(action_choices)

   observation,reward,done,info = env.step(action)

   if env.is_touching():
     observations.append(observation)
     actions.append(action)
     touch_count += 1
     print("is touching")
   if touch_count >= 20:
     print("WINNER!!")
     winners += 1
     touch_count = 0
     training_sets.append([observations,actions])
     observations = []
     actions = []
     step = 0
     episode +=1
     plan_step = 0
     env.reset()
   elif step >= 100:
     print("closing episode,touch_count",touch_count)
     touch_count = 0
     if len(observations) > 1:
       training_sets.append([observations,actions])
     observations = []
     actions = []
     step = 0
     episode +=1
     plan_step = 0
     env.reset()
   if step % 5 == 0:
     print("episode",episode,"step", step,"plan", plan,"plan_step",plan_step,"touch_count",touch_count,"winners",winners)
   step +=1 

