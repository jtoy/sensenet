import sys, argparse, random, glob
import sensenet
import numpy as np
import tensorflow as tf
from cnnModel import cnn_model_fn
from dqnModel import DeepQNetwork
from collections import Counter
import time

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='SenseNet actor-critic example')
    parser.add_argument('--movement_mode',  default='static_policy', metavar='G', help='dqn, static_policy')
    parser.add_argument('--environment','-e',  default='TouchWandLessRewardEnv', metavar='G')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N', help='batch size (default: 16)')    
    parser.add_argument('--num_classes', type=int)
    parser.add_argument('--num_epochs', type=int, default=None, help='Number of Epochs to run training of CNN. None (blank, default) will run until convergence.')
    parser.add_argument('--num_steps', type=int, default=10000, help='Number of steps to run the training of CNN. Default is 10000')
    parser.add_argument('--render', action='store_true', help='render the environment')
    parser.add_argument('--lr',type=float,  default=0.1)    
    parser.add_argument('--model_path', type=str, default='tmp/CNN/', help='path to store/retrieve CNN model')
    parser.add_argument('--log_file', type=str, default='cnn_log.txt', help='file to store logging information')
    parser.add_argument('--agent_path', type=str, default='tmp/DQN/dqn.ckpt', help='path to store/retrieve agent')
    parser.add_argument('--mode', type=str, default="all", help='train/test/all model')
    parser.add_argument('--data_path', type=str, default="data/touchnet_v2/")    
    parser.add_argument('--name', type=str)    
    parser.add_argument('--tensorboard', type=str)    
    parser.add_argument('--test_split_ratio', type=float, default=None)
    args = parser.parse_args()

    env = sensenet.make(args.environment,vars(args))

    if args.name != None:
        args.model_path = "models/"+args.name
        args.log_path = "logs/"+args.name
        args.tensorboard = args.name

    if args.num_classes is None:
        args.num_classes = env.classification_n()

    game_length = env.max_steps      

    if args.environment == 'TouchWandLessRewardEnv' or args.environment == 'TouchWandEnv-v0':
        num_cols = 10000 # number of features in the observation
    elif args.environment == 'FingerJointEnv':
        num_cols = 75000

    cnn_features_TD = np.zeros((len(env.train_files)+1, num_cols), dtype=np.float16)
    cnn_labels_TD = np.zeros(len(env.train_files)+1, dtype=np.int32)

    # cnn arrays for evaluation phase
    cnn_features_ED = np.zeros((len(env.test_files)+1, num_cols), dtype=np.float16)
    cnn_labels_ED = np.zeros(len(env.test_files)+1, dtype=np.int32)

    TD_cnt = 0 # counter to keep track of how many times we touch in the training phase
    ED_cnt = 0 # counter to keep track of how many times we touch in the evaluation phase

    classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, 
                                        model_dir=args.model_path,
                                        params={'num_classes':args.num_classes,
                                                'environment': args.environment})
    if args.movement_mode == 'dqn':
        mem_size = 10*game_length # not important because we're not learning or storing transitions
        game_length = env.max_steps 
        if args.environment == 'TouchWandLessRewardEnv' or args.environment == 'TouchWandEnv':
            observation_size = 10000              
        elif args.environment == 'FingerJointEnv':
            observation_size = 75000
        
        e_greedy_inc = 0.05/game_length
        RL = DeepQNetwork(n_actions=env.action_space.n,
                n_features=observation_size,
                learning_rate=args.lr, e_greedy=0.9,
                replace_target_iter=100, memory_size=mem_size,
                e_greedy_increment=e_greedy_inc,
                model_dir=args.agent_path)
        RL.loadModel()
    start_time = time.time()
    if args.mode == "train" or args.mode == "all":        
        num_files = len(env.train_files)+1
        ep_r = np.zeros(num_files)
        ep_touch = np.zeros(num_files)
        i_episode = 0
        num_games = len(env.train_files)
        while not env.done_training:
            observation = env._reset(mode='train-all')
            done = False
            print('\nstart episode ', i_episode)
            while not done:  
                if args.movement_mode == "static_policy":                  
                    observation_, reward, done, info = env.step(0) # + x direction
                elif args.movement_mode == 'dqn':
                    action = RL.choose_action(observation)
                    observation_, reward, done, info = env.step(action)
                if env.is_touching():
                    print('\nIn episode ', i_episode, 'touching at step', env.steps)

                    cnn_features_TD[TD_cnt] = observation_
                    cnn_labels_TD[TD_cnt] = env.class_label
                    TD_cnt += 1
                    ep_touch[i_episode] += 1

                    if TD_cnt % args.batch_size == 0:
                        print('\n\n\ngetting ready to train classifier. Total touches: ', 
                                np.sum(ep_touch), '\n\n\n')
                        current_batch = int(np.sum(ep_touch))
                        last_batch = current_batch - int(args.batch_size)
                        train_input_fn = tf.estimator.inputs.numpy_input_fn(
                                x={"x": cnn_features_TD[last_batch:current_batch]},
                                y=cnn_labels_TD[last_batch:current_batch],
                                batch_size=args.batch_size,
                                num_epochs=args.num_epochs,
                                shuffle=True)

                        classifier.train(input_fn=train_input_fn,
                                steps=args.num_steps,
                                hooks=None)

                        print('\nclassifier trained\n')                    
                        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                                x={"x": cnn_features_TD[last_batch:current_batch]},
                                y=cnn_labels_TD[last_batch:current_batch],
                                num_epochs=1,
                                shuffle=False)

                        eval_results = classifier.evaluate(input_fn=eval_input_fn)
                        print('These are the results of my evaluations')
                        print(eval_results)
                        print('\n\n')
                        pred_input_fn = tf.estimator.inputs.numpy_input_fn(
                                x={"x": cnn_features_TD[last_batch:current_batch]},
                                num_epochs=1,
                                shuffle=False)
                        pred_results = list(classifier.predict(input_fn=pred_input_fn))
                        predicted_classes = [p["classes"] for p in pred_results]
                        for i, classes in enumerate(predicted_classes):
                            print('I predicted ', classes, 'actual class is ', cnn_labels_TD[last_batch:current_batch])

                        with open(args.log_file, 'a') as f:
                            f.write('Training Phase  batch number ')
                            batch = int(current_batch / args.batch_size)
                            f.write(str(batch))
                            f.write('\tevaluation results ')
                            f.write(str(eval_results))
                            f.write('\nCommand line arguments\t')
                            f.write(str(vars(args)))
                            f.write('\nSanity Check\npredicted classes\n')
                            f.write(str(predicted_classes))
                            f.write('\nactual classes\n')
                            f.write(str(cnn_labels_TD[last_batch:current_batch]))
                            f.write('\n\n\n')

                        exporter = tf.estimator.Exporter()
                        data = exporter.export(estimator=classifier,
                                               export_path=args.model_path,
                                               checkpoint_path=args.model_path,
                                               eval_result=eval_results,
                                               is_the_final_export=False)                            
                observation = observation_                
            i_episode += 1
        print('---------------')
        print('out of ', i_episode+1, 'games we touched ', np.sum(ep_touch), 'times')
        
    data = exporter.export(estimator=classifier,
                            export_path=args.model_path,
                            checkpoint_path=args.model_path,
                            eval_result=eval_results,
                            is_the_final_export=True)

    if args.mode == "all" or args.mode == 'test':        
        num_games = len(env.test_files)
        for i_episode in range(num_games):
            done = False
            observation = env._reset(mode='test')
            while not done:
                if args.movement_mode == "static_policy":                  
                    observation_, reward, done, info = env.step(0) # + x direction
                elif args.movement_mode == 'dqn':
                    action = RL.choose_action(observation)
                    observation_, reward, done, info = env.step(action)                                              
                if env.is_touching():
                    print('\n\nin episode: ', i_episode, 'touching object and storing data to evaluate \n\n')
                    cnn_features_ED[ED_cnt] = observation_
                    cnn_labels_ED[ED_cnt] = env.class_label
                    ED_cnt += 1
                    if ED_cnt % args.batch_size == 0:
                        current_batch = int(ED_cnt)
                        last_batch = current_batch - int(args.batch_size)
                        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                                                    x={"x": cnn_features_ED[last_batch:current_batch]},
                                                    y=cnn_labels_ED[last_batch:current_batch],
                                                    num_epochs=1,
                                                    shuffle=False)
                        eval_results = classifier.evaluate(input_fn=eval_input_fn)

                        print('These are the results of my evaluations')
                        print(eval_results)
                        print('\n--------------------\n')
                
                        pred_input_fn = tf.estimator.inputs.numpy_input_fn(
                                x={"x": cnn_features_ED[last_batch:current_batch]},
                                num_epochs=1,
                                shuffle=False)
                        pred_results = list(classifier.predict(input_fn=pred_input_fn))
                        predicted_classes = [p["classes"] for p in pred_results]
                        for i, classes in enumerate(predicted_classes):
                            print('I predicted ', classes, 'actual class is ', cnn_labels_ED[last_batch:current_batch])

                        with open(args.log_file, 'a') as f:
                            f.write('EvaluationPhase  batch number ')
                            batch = int(current_batch / args.batch_size)
                            f.write(str(batch))
                            f.write('\tevaluation results ')
                            f.write(str(eval_results))
                            f.write('\nCommand line arguments\t')
                            f.write(str(vars(args)))
                            f.write('\nSanity Check\npredicted classes\n')
                            f.write(str(predicted_classes))
                            f.write('\nactual classes\n')
                            f.write(str(cnn_labels_ED[last_batch:current_batch]))
                            f.write('\n\n\n')

                observation = observation_ 
    end_time = time.time()
    with open(args.log_file, 'a') as f:
        f.write('\nTotal elapsed time (seconds)\t')
        f.write(str(end_time-start_time))
