
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


parser = argparse.ArgumentParser(description='SenseNet reinforce example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--epsilon', type=float, default=0.6, metavar='G', help='epsilon value for random action (default: 0.6)')
parser.add_argument('--seed', type=int, default=42, metavar='N', help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='interval between training status logs (default: 10)') 
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--debug', action='store_true', help='debug')
parser.add_argument('--gpu', action='store_true', help='use GPU')
parser.add_argument('--log', type=str, help='log experiment to tensorboard')
parser.add_argument('--model_path', type=str, help='path to store/retrieve model at')
parser.add_argument('--data_path', type=str,default='./objects', help='path to training data')
parser.add_argument('--name', type=str, help='name for logs/model')
parser.add_argument('--mode', type=str, default="train", help='train/test/all model')
parser.add_argument('--num_episodes', type=int, default=1000, help='number of episodes')
parser.add_argument('--obj_type', type=str, default="stl", help='obj or stl')
args = parser.parse_args()

if args.name:
  model_path = "./models/"+args.name
  log_name = args.name
else:
  if args.log:
    log_name = args.log
  else:
    log_name = None
  if args.model_path:
    model_path = args.model_path
  else:
    model_path = None
# Command to visualize logs: 
#   tensorboard --logdir runs
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


class CNNLSTM(nn.Module):

  def __init__(self, classification_n):
    super(CNNLSTM, self).__init__()

    # CNN
    self.layer1 = nn.Sequential(
      nn.Conv2d(1, 16, kernel_size=5, padding=2),
      nn.BatchNorm2d(16),
      nn.MaxPool2d(2),
      nn.Conv2d(16, 32, kernel_size=5, padding=2),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.MaxPool2d(2))
    self.layer2 = nn.Sequential(
      nn.Conv2d(32, 64, kernel_size=5, padding=2),
      nn.BatchNorm2d(64),
      nn.MaxPool2d(2),
      nn.Conv2d(64, 128, kernel_size=5, padding=2),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.MaxPool2d(2))
    self.layer3 = nn.Sequential(
      nn.Conv2d(128, 256, kernel_size=3, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(),
      nn.MaxPool2d(2))
    self.conv_feature_map_size = 6*6*256

    # LSTM
    self.rnn_hidden_size = 40
    self.rnn = nn.LSTM(
      input_size=self.conv_feature_map_size, 
      hidden_size=self.rnn_hidden_size
    )

    # OUTPUT
    self.out_linear = nn.Linear(self.rnn_hidden_size, classification_n)

  def forward(self, x):
    x = x.unsqueeze(1).float()

    # CNN
    cnn_h = self.layer1(x)
    # print("size after layer1:", cnn_h.size())
    cnn_h = self.layer2(cnn_h)
    # print("size after layer2:", cnn_h.size())
    cnn_h = self.layer3(cnn_h)
    # print("size after layer3:", cnn_h.size())

    # LSTM
    rnn_inputs = cnn_h.view(cnn_h.size(0), -1)  # Flatten CNN feature map
    rnn_hidden = (
      autograd.Variable(torch.randn((1, 1, self.rnn_hidden_size)), requires_grad=False),
      autograd.Variable(torch.randn((1, 1, self.rnn_hidden_size)), requires_grad=False)
    )  # Init first time step in RNN
    for inp in rnn_inputs:
      # Only the last output from the LSTM is kept. 
      # We use a many-to-one RNN architecture. 
      rnn_out, rnn_hidden = self.rnn(inp.view(1, 1, -1), rnn_hidden)
    # print("size of rnn_out:", rnn_out.size())
    rnn_out = rnn_out[0]

    # OUTPUT
    return self.out_linear(rnn_out)  # Linear for good classification output size


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

def finish_episode_learning(model, optimizer):
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


# Training:

env = SenseEnv(vars(args))
print("action space: ",env.action_space())
print("class count: ",env.classification_n())
model = Policy(env.observation_space(),env.action_space_n())
cnn_lstm = CNNLSTM(env.classification_n())
if args.gpu and torch.cuda.is_available():
  model.cuda()
  cnn_lstm.cuda()
if model_path:
  if os.path.exists(model_path+"/model.pkl"):
    print("loading pretrained models")
    model.load_state_dict(torch.load(model_path+"/model.pkl"))
    cnn_lstm.load_state_dict(torch.load(model_path+"/cnn_lstm.pkl"))

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

classifier_criterion = nn.CrossEntropyLoss()
classifier_optimizer = torch.optim.Adam(cnn_lstm.parameters(), lr=0.001)

running_reward = 10
total_steps = 0
max_steps = 500
touched_episodes = 0
  
if args.mode == "train" or args.mode == "all":

  for i_episode in range(0,args.num_episodes):
    # New object (aka new episode)
    observation = env.reset()
    average_activated_pixels = []
    observed_touches = []
    print("episode:", i_episode)

    for step in range(max_steps):
      # Move and touch the object in this loop. Record touches only as inputs.
      action = select_action(observation,env.action_space_n(),args.epsilon)
      observation, reward, done, info = env.step(action)
      model.rewards.append(reward)
      average_activated_pixels.append(np.mean(observation))
      if env.is_touching():
        observed_touches.append(observation.reshape(200,200))
      if done:
        break

    if len(observed_touches) != 0:
      touched_episodes += 1
      print("touches in episode ", i_episode)
    if 1==2 and len(observed_touches) != 0:
      # If touched, train classifier. The touched sequence is sent in a CNN LSTM.
      print("  >> {} touches in current episode <<".format(len(observed_touches)))

      observed_touches = torch.LongTensor(torch.from_numpy(np.asarray(observed_touches)))
      print("  env.class_label:", env.class_label)
      label = torch.from_numpy(np.asarray([env.class_label]))
      if args.gpu and torch.cuda.is_available():
        observed_touches = observed_touches.cuda()
        label = label.cuda()
      observed_touches = Variable(observed_touches)
      label = Variable(label)

      # Classifier is learning:
      classifier_optimizer.zero_grad()
      output = cnn_lstm(observed_touches)  # Prediction
      # print("output:", output.data)
      # print("label:", label.data)
      loss = classifier_criterion(output, label)
      loss.backward()
      classifier_optimizer.step()
      print ('  Loss: %.4f' %(loss.data[0]))

      if log_name:
        writer.add_scalar(log_name + "/loss",loss.data[0],total_steps)

    running_reward = running_reward * 0.99 + step * 0.01
    total_steps +=1
    print("  learning...")
    finish_episode_learning(model, optimizer)

    if log_name:
      writer.add_scalar(log_name+"/reward",running_reward,total_steps)
      writer.add_scalar(log_name+"/touches",len(observed_touches),total_steps)
      writer.add_scalar(log_name+"/average_activated_pixels",np.mean(average_activated_pixels),total_steps)
    if i_episode % args.log_interval == 0:
      print('  Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(i_episode, step, running_reward))
    if running_reward > 5000: #env.spec.reward_threshold:
      print("  Solved! Running reward is now {} and the last episode runs to {} time steps!".format(running_reward, t))
      break
    if model_path:
      env.mkdir_p(model_path)
      torch.save(model.state_dict(), os.path.join(model_path, 'policy.pkl' ))
      torch.save(model.state_dict(), os.path.join(model_path, 'cnn_lstm.pkl' ))
  print("touched ", touched_episodes, " times")

elif args.mode == "test" or args.mode == "all":
  #test
  test_labels = []
  predicted_labels = []
  steps_to_guess = []
  correct = 0
  total_correct = 0
  total = 0
  touched_episodes = 0

  for i_episode in range(100):
    # New object (aka new episode)
    observation = env.reset()
    observed_touches = []
    print("episode:", i_episode)

    for step in range(max_steps):
      # Move and touch the object in this loop. Record touches only as inputs.
      action = select_action(observation,env.action_space_n(),args.epsilon)
      observation, reward, done, info = env.step(action)
      model.rewards.append(reward)
      if env.is_touching():
        observed_touches.append(observation.reshape(200,200))
      if done:
        break

    if len(observed_touches) != 0:
      observed_touches = torch.LongTensor(torch.from_numpy(np.asarray(observed_touches)))
      touched_episodes += 1
      if args.gpu and torch.cuda.is_available():
        observed_touches = observed_touches.cuda()
      observed_touches = Variable(observed_touches)

      # Predicting and evaluation
      output = cnn_lstm(observed_touches)
      prob, predicted = torch.max(output.data, 1)
      print("predicted", predicted[0][0])
      correct = int(predicted[0][0] == env.class_label)
      total_correct += correct
      print("  predicted ", predicted[0][0], " with prob ", prob[0], " correct answer is: ", env.class_label)
    else:
      print("  no touches!")
    total += 1

  print('Accuracy of the network: %d %%' % (100 * total_correct / total )) 
  print('touched ',  touched_episodes, ' times') 

else:
  for i_episode in range(100):
    observation = env.reset()
    for step in range(1000):
      env.render()
      action = np.random.choice(env.action_space_n())
      observation,reward,done,info = env.step(action)
      print(observation)
