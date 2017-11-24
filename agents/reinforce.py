
import sys
sys.path.append('..')

from env import SenseEnv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import numpy as np

from itertools import count
from collections import namedtuple
import time,os,math,inspect,re,random,argparse

#tensorboard --logdir runs

writer = SummaryWriter()
SavedAction = namedtuple('SavedAction', ['action', 'value'])
class Policy(nn.Module):
  def __init__(self,observation_space_n,action_space_n):
    super(Policy, self).__init__()
    self.affine1 = nn.Linear(observation_space_n, 256)
    self.action1 = nn.Linear(256, 128)
    self.value1 = nn.Linear(256, 128)
    self.action_head = nn.Linear(128, action_space_n)
    self.value_head = nn.Linear(128, 1)
    self.saved_actions = []
    self.rewards = []
    self.init_weights()

  def init_weights(self):
    self.affine1.weight.data.uniform_(-0.1, 0.1)
    self.action1.weight.data.uniform_(-0.1, 0.1)
    self.action_head.weight.data.uniform_(-0.1, 0.1)
    self.value1.weight.data.uniform_(-0.1, 0.1)

  def forward(self, x):
    if args.gpu and torch.cuda.is_available():
      # TODO: clean way to access "args.gpu" from here.
      x = x.cuda()
    x = F.relu(self.affine1(x))
    xa = F.relu(self.action1(x))
    xv = F.relu(self.value1(x))
    action_scores = self.action_head(xa)
    state_values = self.value_head(xv)
    return F.softmax(action_scores)
    #return F.softmax(action_scores), state_values

class CNN(nn.Module):
  def __init__(self,classification_n):
    super(CNN, self).__init__()
    self.layer1 = nn.Sequential(
      nn.Conv2d(1, 16, kernel_size=5, padding=2),
      nn.BatchNorm2d(16),
      nn.ReLU(),
      nn.MaxPool2d(2))
    self.layer2 = nn.Sequential(
      nn.Conv2d(16, 32, kernel_size=5, padding=2),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.MaxPool2d(2))
    #self.fc = nn.Linear(7*7*32, 2)
    self.fc = nn.Linear(80000, classification_n)
      
  def forward(self, x):
    x = x.unsqueeze(1).float()
    out = self.layer1(x)
    out = self.layer2(out)
    #print("size before",out.size())
    out = out.view(out.size(0), -1)
    #print("size after",out.size())
    out = self.fc(out)
    return out

parser = argparse.ArgumentParser(description='SenseNet reinforce example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--epsilon', type=float, default=0.6, metavar='G', help='epsilon value for random action (default: 0.6)')
parser.add_argument('--seed', type=int, default=42, metavar='N', help='random seed (default: 42)')
parser.add_argument('--batch_size', type=int, default=42, metavar='N', help='batch size (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='interval between training status logs (default: 10)') 
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--debug', action='store_true', help='debug')
parser.add_argument('--gpu', action='store_true', help='use GPU')
parser.add_argument('--log', type=str, help='log experiment to tensorboard')
parser.add_argument('--model_path', type=str, help='path to store/retrieve model at')
parser.add_argument('--mode', type=str, default="train", help='train/test/all model')
args = parser.parse_args()


def select_action(state,n_actions,epsilon=0.2):
  if np.random.rand() < epsilon:
    return np.random.choice(n_actions)
  else:
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = model(Variable(state))
    action = probs.multinomial()
    #model.saved_actions.append(SavedAction(action, state_value))
    model.saved_actions.append(action)
    return action.data[0][0]


def finish_episode():
  R = 0
  rewards = []
  for r in model.rewards[::-1]:
    R = r + args.gamma * R
    rewards.insert(0, R)
  rewards = torch.Tensor(rewards)
  rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
  for action, r in zip(model.saved_actions, rewards):
    action.reinforce(r)
  optimizer.zero_grad()
  autograd.backward(model.saved_actions, [None for _ in model.saved_actions])
  optimizer.step()
  del model.rewards[:]
  del model.saved_actions[:]


#train
env = SenseEnv(vars(args))
print("action space: ",env.action_space())
model = Policy(env.observation_space(),env.action_space_n())
cnn = CNN(env.classification_n())
if args.gpu and torch.cuda.is_available():
  env.gpu()
  model.cuda()
  cnn.cuda()
if args.model_path:
  if os.path.exists(args.model_path+"/model.pkl"):
    print("loading pretrained models")
    model.load_state_dict(torch.load(args.model_path+"/model.pkl"))
    cnn.load_state_dict(torch.load(args.model_path+"/cnn.pkl"))

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

classifier_criterion = nn.CrossEntropyLoss()
classifier_optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)

running_reward = 10
batch = []
labels = []
total_steps = 0
if args.mode == "train" or args.mode == "all":
  for i_episode in range(0,1000):
    observation = env.reset()
    touch_count = 0
    average_activated_pixels = []
    print("episode: ", i_episode)
    for step in range(1000):
      action = select_action(observation,env.action_space_n(),args.epsilon)
      observation, reward, done, info = env.step(action)
      model.rewards.append(reward)
      average_activated_pixels.append(np.mean(observation))
      
      if env.is_touching():
        touch_count += 1
        #print(observation)
        #time.sleep(3)
        print("touching!")
        #print("batch size", len(batch))
        if len(batch) > args.batch_size:
          #TODO GPU support
          #batch = torch.from_numpy(np.asarray(batch))
          batch = torch.LongTensor(torch.from_numpy(np.asarray(batch)))
          labels = torch.from_numpy(np.asarray(labels))
          #labels = torch.LongTensor(torch.from_numpy(np.asarray(labels)))
          if args.gpu and torch.cuda.is_available():
            batch = batch.cuda()
            labels = labels.cuda()
          batch = Variable(batch)
          labels = Variable(labels)
          classifier_optimizer.zero_grad()
          outputs = cnn(batch)
          loss = classifier_criterion(outputs, labels)
          loss.backward()
          classifier_optimizer.step()
          print ('Loss: %.4f' %(loss.data[0]))
          if args.log:
            writer.add_scalar(args.log + "/loss",loss.data[0],total_steps)
          batch = []
          labels = []
        else:
          batch.append(observation.reshape(200,200))
          labels.append(env.class_label)
      if done:
        break
    running_reward = running_reward * 0.99 + step * 0.01
    total_steps +=1
    print("learning...")
    finish_episode()

    if args.log:
      writer.add_scalar(args.log+"/reward",running_reward,total_steps)
      writer.add_scalar(args.log+"/touches",touch_count,total_steps)
      writer.add_scalar(args.log+"/average_activated_pixels",np.mean(average_activated_pixels),total_steps)
    if i_episode % args.log_interval == 0:
      print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(i_episode, step, running_reward))
    if running_reward > 5000: #env.spec.reward_threshold:
      print("Solved! Running reward is now {} and the last episode runs to {} time steps!".format(running_reward, t))
      break
    if args.model_path:
      env.mkdir_p(args.model_path)
      torch.save(model.state_dict(), os.path.join(args.model_path, 'policy.pkl' ))
      torch.save(model.state_dict(), os.path.join(args.model_path, 'cnn.pkl' ))
elif args.mode == "test" or args.mode == "all":
  #test
  test_labels = []
  predicted_labels = []
  steps_to_guess = []
  correct = 0
  total_correct = 0
  total = 0
  max_steps = 500
  for i_episode in range(100):
    guesses = []
    print("testing on a new object")
    observation = env.reset()
    print("episode ",i_episode)
    for step in range(max_steps):
      action = select_action(observation,env.action_space_n(),args.epsilon)
      observation, reward, done, info = env.step(action)
      model.rewards.append(reward)
      #if confidence over 90%, then use it
      #if (t >= max_steps-1 and len(guesses) == 0) or env.is_touching:
      if env.is_touching():
        x = [observation.reshape(200,200)]
        x = torch.LongTensor(torch.from_numpy(np.asarray(x)))
        x = Variable(x)
        output = cnn(x)
        prob, predicted = torch.max(output.data, 1)
        correct = int(predicted[0][0] == env.class_label)
        total_correct += correct
        total += 1
        print("predicted ", predicted[0][0], " with prob ", prob[0][0], " correct answer is: ",env.class_label)

        break
  print('Accuracy of the network: %d %%' % (100 * total_correct / total )) 
else:
  for i_episode in range(100):
    observation = env.reset()
    for step in range(1000):
      env.render()
      action = np.random.choice(env.action_space_n())
      observation,reward,done,info = env.step(action)
      print(observation)
  
  
