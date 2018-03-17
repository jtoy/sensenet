import argparse
import sensenet
import numpy as np
import tensorflow as tf

import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from dqnModel import DeepQNetwork
from cnnModel import cnn_model_fn, cnn_lstm_model_fn
from collections import Counter

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='SenseNet actor-critic example')
    parser.add_argument('--environment','-e',  default='TouchWandLessRewardEnv', metavar='G')        
    parser.add_argument('--render', action='store_true', help='render the environment')    
    parser.add_argument('--data_path', type=str, default="data/touchnet_v2")    
    parser.add_argument('--agent_path', type=str, default='tmp/DQN/dqn.ckpt', help='path to store/retrieve agent')
    parser.add_argument('--log_file', type=str, default="logs/dqn.log")    
    parser.add_argument('--tensorboard', type=str)    
    parser.add_argument('--name', type=str)    
    parser.add_argument('--lr',type=float,  default=0.1)    
    parser.add_argument('--test_split_ratio', type=float, default=None)
    parser.add_argument('--num_games', type=int, default=100, help='number of games to train and evaluate')
    parser.add_argument('--mem_size', type=int, default=10, help='total number of games to remember. Keep in mind the Finger consumes huge amounts of memory')

    args = parser.parse_args()
    if args.name != None:
        args.agent_path = "models/"+args.name
        args.log_path = "logs/"+args.name
        args.tensorboard = args.name


    env = sensenet.make(args.environment, vars(args))

    game_length = env.max_steps
    e_greedy_inc = 0.05/game_length
    training_touches = 0
    num_games = args.num_games
    total_steps = 0
    ep_r = np.zeros(num_games)
    ep_length = np.zeros(num_games)
    if args.environment == 'TouchWandLessRewardEnv' or args.environment == 'TouchWandEnv':
        observation_size = 10000
        mem_size = args.mem_size*game_length
    elif args.environment == 'FingerJointEnv':
        observation_size = 75000
        mem_size = args.mem_size*game_length
    RL = DeepQNetwork(n_actions=env.action_space.n,
            n_features=observation_size,
            learning_rate=args.lr, e_greedy=0.9,
            replace_target_iter=100, memory_size=mem_size,
            e_greedy_increment=e_greedy_inc,
            model_dir=args.agent_path)

    for i_episode in range(num_games):
        observation = env._reset(mode='train-all')

        done = False
        while env.steps < game_length and not done:
            action = RL.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            RL.store_transition(observation, action, reward, observation_)
            ep_r[i_episode] += reward
            if total_steps > 1000:
                cost = RL.learn()    

            if (env.steps % 500 == 0):
                    print('\nepisode: ', i_episode, 'step: ', env.steps,
                          'episode reward ', ep_r[i_episode])

            if env.is_touching():
                print('\nIn episode ', i_episode, 'touching at step',
                        env.steps, 'total reward is ', ep_r[i_episode]) 
                training_touches += 1               
            total_steps += 1
        ep_length[i_episode] = env.steps
    RL.saveModel()
    
    fig = plt.figure()
    ax = fig.add_subplot(211, label="Total Reward")
    ax2 = fig.add_subplot(212, label="Session Length")
    x = [i for i in range(num_games)]
    ax.plot(x, ep_r, 'r-')
    ax.set_xlabel('Episode', color='black')
    ax.set_ylabel('Total Reward', color='red')
    ax.tick_params(axis='x', colors='red')
    ax.tick_params(axis='y', colors='red')

    ax2.plot(x, ep_length, 'b-')
    ax2.set_xlabel('Episode', color='black')
    ax2.set_ylabel('Episode Length', color='blue')
    #ax2.yaxis.tick_left()
    ax2.yaxis.set_label_position('left')
    ax2.tick_params(axis='y', colors='blue')
    #plt.show()
    fig_name = str(args.num_games) + 'games' + str(args.mem_size) + 'mem_size' + str(args.lr) + 'learning_rate' + '.png'
    fig.savefig("graphs/"+str(args.tensorboard)+".png")
    

    print('\ntraining is finished. Moving to evaluation phase. Will no longer call RL.learn\n')
    ep_r = np.zeros(num_games)
    total_touches = 0
    for i_episode in range(num_games):
        observation = env._reset(mode='train-all')
        done = False
        while env.steps < game_length and not done:
            action = RL.choose_action(observation)
            observation_, reward, done, info = env.step(action)            
            ep_r[i_episode] += reward            
            if (env.steps % 500 == 0):
                    print('\nepisode: ', i_episode, 'step: ', env.steps,
                          'episode reward ', ep_r[i_episode])

            if env.is_touching():
                print('\nIn episode ', i_episode, 'touching at step',
                        env.steps, 'total reward is ', ep_r[i_episode])
                total_touches += 1
            total_steps += 1

    print('In the evaluation phase, out of ', num_games, 'games, we touched', total_touches, 'times')
    print('In the training phase, out of ', num_games, 'games, we touched', training_touches, 'times')
