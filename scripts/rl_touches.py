import sys,random,glob,argparse,uuid,math
sys.path.append('..')
import sensenet
import pybullet as p
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

forward = 3
left = 0
right = 1
up = 4
down = 5
indexup = 6
indexdown = 7
action_choices = [forward,left,right,up,down] #dont move index for now

plans = [([left]*5),([right] * 5),([up] * 5),([down] * 5),([indexup] * 5),([indexdown] * 5),] + (["random"] * 6)
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

class Policy(nn.Module):

  def __init__(self,observation_space_n,action_space_n):
    super(Policy, self).__init__()
    self.affine1 = nn.Linear(observation_space_n, 128)
    self.affine2 = nn.Linear(128, action_space_n)
    self.saved_log_probs = []
    self.rewards = []
    #self.init_weights()

  def init_weights(self):
    self.affine1.weight.data.uniform_(-0.1, 0.1)
    self.affine2.weight.data.uniform_(-0.1, 0.1)

  def forward(self, x):
    #if args.gpu and torch.cuda.is_available():
      # TODO: clean way to access "args.gpu" from here.
    #  x = x.cuda()
    x = F.relu(self.affine1(x))
    action_scores = self.affine2(x)
    return F.softmax(action_scores,dim=1)
 
def select_action(state,n_actions,steps,epsilon=0.2):
  if False and steps < 100:
    return np.random.choice(n_actions)
  else:
    state = torch.from_numpy(state).float().unsqueeze(0)
    print(state)
    probs = policy(Variable(state))
    action = probs.multinomial()
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.data[0]
def finish_episode():
  R = 0
  policy_loss = []
  rewards = []
  for r in policy.rewards[::-1]:
    R = r + args.gamma * R
    rewards.insert(0, R)
  rewards = torch.Tensor(rewards)
  rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
  for log_prob, reward in zip(policy.saved_log_probs, rewards):
    policy_loss.append(-log_prob * reward)
  optimizer.zero_grad()
  policy_loss = torch.cat(policy_loss).sum()
  policy_loss.backward()
  optimizer.step()
  del policy.rewards[:]
  del policy.saved_log_probs[:]
parser = argparse.ArgumentParser()
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--environment', type=str, default="HandEnv-v0")
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
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
policy = Policy(3,env.action_space_n())
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
total_step = 0
running_reward = 10
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
  prev_distance = 10000000
  for epoch in range(args.epochs):
    env.reset()
    while(1):
      points = p.getClosestPoints(env.obj_to_classify,env.agent,10000000,-1,22)
      al = points[0][7]
      ol = points[0][6]
      xd = (al[0]-ol[0])/2
      yd = (al[1]-ol[1])/2
      zd = (al[2]-ol[2])/2
      state = np.asarray([xd,td,zd])
      #print("xd",xd,"yd",yd,"zd",zd:wu)
      learning = False

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
          action = select_action(np.asarray([xd,yd,zd]),env.action_space_n(),total_step)
          learning = True

      observation,reward,done,info = env.step(action)
      distance = math.sqrt(xd**2+yd**2+zd**2)
      if distance < prev_distance:
        reward += 1 
        #reward += 1 * (max_steps - self.steps)
      policy.rewards.append(reward)
      prev_distance = distance
      print(len(policy.rewards))
      #if total_step % 20 == 0:
      #if total_step > 200 and total_step % 50 == 0:
      if learning == True and total_step % 50 == 0:
        print("episodeDDDDD")
        finish_episode()

      if env.is_touching():
        print("touch")

        observations.append(observation.reshape(200,200))
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
      elif step >= 100:
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
      total_step +=1
      if args.fast_exit != 0 and episode >= args.fast_exit:
          sys.exit()
      if step % 20 == 0:
        print("episode",episode,"label",label,"step", step,"plan", plan,"plan_step",plan_step,"touch_count",touch_count,"winners",winners)
      step +=1
