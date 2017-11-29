import sys
sys.path.append('..')
from env import SenseEnv
import pybullet as pb
obj_file = '../tests/data/cube.stl'
action_plan = [{'moves':14,'action':3}] #full frontal of cube
#action_plan = [{'moves':14,'action':3},{'moves':11,'action':4}] #top edge,vertical line

obj_file = '../tests/data/sphere.stl'
action_plan = [{'moves':20,'action':3}]  #sphere circular area

obj_file = '../tests/data/pyramid.stl'
action_plan = [{'moves':8,'action':3}]  #pyramid angle
action_plan = [{'moves':8,'action':3},{'moves':10,'action':4},{'moves':5,'action':3}]  #mid pyramid
action_plan = [{'moves':5,'action':1},{'moves':10,'action':3},{'moves':15,'action':4},{'moves':10,'action':3}]  #top of pyramid
if len(sys.argv) == 2:
  obj_file = sys.argv[1]
  print("loading",obj_file)
env = SenseEnv({'render':True,'debug':True,'obj_path':obj_file})
action_plan_counter  = 0
action_step = 0

def max_action_plan_steps(action_plan):
  return sum([x['moves'] for x in action_plan ])

def process_action_plan(action_plan,action_plan_counter=0,action_step=0):
  max_steps = max_action_plan_steps(action_plan)
  if action_step == 0:
    start_for_current_action = 0
  else:
    start_for_current_action = sum([x['moves'] for x in action_plan[:action_step]])
  if action_plan_counter-start_for_current_action >= action_plan[action_step]['moves']:
    action_step += 1
  #print("before while apc ac start moves",action_plan_counter, action_step,start_for_current_action)
  while action_plan_counter < max_steps:
    action_plan_counter += 1
    return action_plan[action_step]['action'],action_plan_counter,action_step

while (1):
  n = -1
  if action_plan_counter < max_action_plan_steps(action_plan):
    n,action_plan_counter,action_step = process_action_plan(action_plan,action_plan_counter,action_step)
  else:
    key = pb.getKeyboardEvents()
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
    #      print("new number",n)
        else:
          print("key: ",k)
  if n > -1:
   observation,reward,done,info = env.step(n)

