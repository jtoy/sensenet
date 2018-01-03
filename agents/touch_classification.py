import sensenet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import glob
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import numpy as np
from itertools import count
from collections import namedtuple
import time,os,math,inspect,re,random,argparse


parser = argparse.ArgumentParser(description='SenseNet reinforce example')
parser.add_argument('--seed', type=int, default=42, metavar='N', help='random seed (default: 42)')
parser.add_argument('--num_epochs', type=int, default=5, metavar='N')
parser.add_argument('--lr', type=float, default=0.001, metavar='G', help='learning rate')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='interval between training status logs (default: 10)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--debug', action='store_true', help='debug')
parser.add_argument('--gpu', action='store_true', help='use GPU')
parser.add_argument('--log', type=str, help='log experiment to tensorboard')
parser.add_argument('--model_path', type=str, help='path to store/retrieve model at')
parser.add_argument('--data_path', type=str,default='./objects', help='path to training data')
parser.add_argument('--name', type=str, help='name for logs/model')
parser.add_argument('--mode', type=str, default="all", help='train/test/all model')
parser.add_argument('--environment', type=str, default="HandEnv-v0")
parser.add_argument('--obj_type', type=str, default="obj", help='obj or stl')
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
writer = SummaryWriter()

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

#env = sensenet.make(args.environment,vars(args))

#action_choices = [forward,left,right,up,down] #dont move index for now
cnn_lstm = CNNLSTM(11)
#cnn_lstm = CNNLSTM(env.classification_n())

if args.gpu and torch.cuda.is_available():
  cnn_lstm.cuda()
if model_path:
  if os.path.exists(model_path+"/cnn_lstm.pkl"):
    print("loading pretrained model")
    cnn_lstm.load_state_dict(torch.load(model_path+"/cnn_lstm.pkl"))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn_lstm.parameters(), lr=args.lr)

class TouchDataset(Dataset):
  def __init__(self, root_dir):
    self.root_dir = root_dir
    self.files = []
    self.labels = []
    self.actions = []
    for filename in glob.iglob(self.root_dir+"/**/*touches.npy", recursive=True):
      label = int(filename.split("/")[-3].split("_")[0])
      #print(filename)
      #print(label)
      action_name = filename.replace("touches","actions")
      self.files.append(filename)
      self.actions.append(action_name)

      self.labels.append(label)
  def __len__(self):
    return len(self.files)
  def __getitem__(self, idx):
    #print(idx)
    touch_path = self.files[idx]
    action_path = self.actions[idx]
    samples = np.load(touch_path)
    label = self.labels[idx]
    actions = np.load(action_path)
    #print(idx,samples.shape)
    #return {'sample':sample,'action':action,'label':label}
    #print(samples,actions,label)
    #print(len(samples))
    #print("label",label)
    return samples,label #actions,label




train_dataset = TouchDataset(args.data_path)
batch_size = 1
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
total_step = len(train_loader)
for epoch in range(10):
  print('epoch=%d -------------------------'%(epoch))
  for i, (touches,label) in enumerate(train_loader):
    touches = touches.squeeze(0).float()
    touches = Variable(touches)
    label = Variable(label)
    optimizer.zero_grad()
    outputs = cnn_lstm(touches)
    loss = criterion(outputs, label)
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
      print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'%(epoch, args.num_epochs, i, total_step,loss.data[0], np.exp(loss.data[0])))

if args.mode == "all" or args.mode == "train":
# Train the Model
  for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
      images = Variable(images)
      labels = Variable(labels)
      # Forward + Backward + Optimize
      optimizer.zero_grad()
      outputs = cnn_lstm(images)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      if (i+1) % 100 == 0:
        print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))
  #load training data

  for i_episode in range(args.num_episodes):
    # New object (aka new episode)
    observation = env.reset()
    average_activated_pixels = []
    touch_count = 0
    observed_touches = []

    for step in range(args.max_steps):
      # Move and touch the object in this loop. Record touches only as inputs.
      observation, reward, done, info = env.step(action)
      model.rewards.append(reward)
      average_activated_pixels.append(np.mean(observation))
      if env.is_touching():
        if args.debug:
          print("touching")
        observed_touches.append(observation.reshape(200,200))
        touch_count += 1
        if touch_count == 1:
          steps_to_first_touch.append(step)
        if args.debug:
          print("touch at step ", step, " in episode ", i_episode)
      #if done:
      #  break

    if len(observed_touches) != 0:
      touched_episodes += 1
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
    print(touch_count, "touches in episode", i_episode)
    finish_episode_learning(model, optimizer)

    if log_name:
      writer.add_scalar(log_name+"/touches",len(observed_touches),total_steps)
      #writer.add_scalar(log_name+"/average_activated_pixels",np.mean(average_activated_pixels),total_steps)
    if i_episode % args.log_interval == 0:
      print('  Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(i_episode, step, running_reward))
    if running_reward > 5000: #env.spec.reward_threshold:
      print("  Solved! Running reward is now {} and the last episode runs to {} time steps!".format(running_reward, t))
      break
    if model_path:
      env.mkdir_p(model_path)
      torch.save(model.state_dict(), os.path.join(model_path, 'cnn_lstm.pkl' ))
