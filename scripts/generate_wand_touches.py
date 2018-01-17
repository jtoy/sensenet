import sys,random,glob,argparse,uuid
sys.path.append('..')
import sensenet
import pybullet as p
import numpy as np

x_plus = 0
x_minus = 1
y_plus = 2
y_minus = 3
z_plus = 4
z_minus =5
forward = x_plus
left = y_plus
right = y_minus
down = z_minus
up = z_plus
action_choices = [forward,left,right,up,down] #dont move index for now
plans = [([left]*5),([right] * 5),([up] * 5),([down] * 5),] + (["random"] * 6)

#plans = [([left]*5),([right] * 5),([up] * 5),([down] * 5),([indexup] * 5),([indexdown] * 5),] + (["random"] * 6)
def random_plan():
  a = []
  for i in range(5):
    a.append(random.randint(0,7))
  return a
plan = None
def save_data(env,label,touches,actions):
  path = "touch_data/"+str(label)+"/"+str(uuid.uuid1())
  env.mkdir_p(path)
  np.save(path+"/touches",touches)
  np.save(path+"/actions",actions)

parser = argparse.ArgumentParser()
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--environment',"-e", type=str, default="TouchWandEnv-v0")
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--folder', type=str)
parser.add_argument('--file', type=str)
parser.add_argument('--fast_exit', type=int, default=0)
args = parser.parse_args()
if args.folder:
  files = list(glob.iglob(args.folder+"/**/*.obj", recursive=True))
else:
  files = [args.file]
random.shuffle(files)
env = sensenet.make(args.environment,{'render':args.render})
for filename in files:
  label = int(filename.split("/")[-3].split("_")[0])
  print(filename)
  print(label)
  path = "touch_data/"+str(label)+"/"
  env._reset({'obj_path':filename})
  env.mkdir_p(path)
  training_sets = []
  observations = []
  actions = []
  touch_count = 0
  step = 0
  episode = 0
  plan_step = 0
  tries = 0
  winners = 0
  past_observation = []
  for epoch in range(args.epochs):
    env.reset()
    while(1):
      points = p.getClosestPoints(env.obj_to_classify,env.agent_mb,10000000,-1,-1)
      #points = p.getClosestPoints(env.obj_to_classify,env.agent,10000000,-1,22)
      al = points[0][7]
      ol = points[0][6]
      #al, _ = p.getBasePositionAndOrientation(env.agent)
      #ol, _ = p.getBasePositionAndOrientation(env.obj_to_classify)
      xd = abs(al[0]-ol[0])/2
      yd = abs(al[1]-ol[1])/2
      zd = abs(al[2]-ol[2])/2
      #print("xd",xd,"yd",yd,"zd",zd)
      #what the hell is 22?

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
        #if zd >= xd:
        if random.random() > 0.5 and zd >= xd:

          action = down
          #print("zd")
        #elif xd >= yd:
        elif random.random() > 0.5 and xd >= yd:

          action = forward
          #print("xd")

        #elif yd >= xd:
        elif random.random() > 0.5 and yd >= xd:
          #print("yd")
          action = left
          #action = right
        else:
          action = random.choice(action_choices)

      observation,reward,done,info = env.step(action)
      if env.is_touching():
        #print("touch")
        print(np.array_equal(observation,past_observation))
        past_observation = observation

        observations.append(observation.reshape(100,100))
        actions.append(action)
        touch_count += 1
        #print("is touching")
      if touch_count >= 20:
        print("WINNER!!")
        winners += 1
        touch_count = 0
        #training_sets.append([observations,actions])
        save_data(env,label,observations,actions)
        observations = []
        actions = []
        step = 0
        episode +=1
        plan_step = 0
        env.reset()
      elif step >= 10000:
        print("closing episode,touch_count",touch_count)
        touch_count = 0
        if len(observations) > 2:
          #training_sets.append([observations,actions])
          save_data(env,label,observations,actions)
        else:
          tries +=1
          if tries > 5:
            print("couldnt get trainig data for item",filename)
            break
        observations = []
        actions = []
        step = 0
        episode +=1
        plan_step = 0
        env.reset()
      if args.fast_exit != 0 and episode >= args.fast_exit:
          sys.exit()
      if step % 20 == 0:
        print("episode",episode,"label",label,"step", step,"plan", plan,"plan_step",plan_step,"touch_count",touch_count,"winners",winners)
      step +=1
