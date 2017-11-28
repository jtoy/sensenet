import sys, argparse
import gym
sys.path.append('..')
from env import SenseEnv
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter


writer = SummaryWriter()
# Deep Q Network off-policy
# got from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/6_OpenAI_gym/RL_brain.py
class DeepQNetwork:
	def __init__(
			self,
			n_actions,
			n_features,
			learning_rate=0.01,
			reward_decay=0.9,
			e_greedy=0.9,
			replace_target_iter=300,
			memory_size=500,
			batch_size=32,
			e_greedy_increment=None,
			output_graph=False,
	):
		self.n_actions = n_actions
		self.n_features = n_features
		self.lr = learning_rate
		self.gamma = reward_decay
		self.epsilon_max = e_greedy
		self.replace_target_iter = replace_target_iter
		self.memory_size = memory_size
		self.batch_size = batch_size
		self.epsilon_increment = e_greedy_increment
		self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

		# total learning step
		self.learn_step_counter = 0

		# initialize zero memory [s, a, r, s_]
		self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

		# consist of [target_net, evaluate_net]
		self._build_net()
		t_params = tf.get_collection('target_net_params')
		e_params = tf.get_collection('eval_net_params')
		self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

		self.sess = tf.Session()

		if output_graph:
			# $ tensorboard --logdir=logs
			# tf.train.SummaryWriter soon be deprecated, use following
			tf.summary.FileWriter("logs/", self.sess.graph)


		self.sess.run(tf.global_variables_initializer())
		self.cost_his = []

	def _build_net(self):
		# ------------------ build evaluate_net ------------------
		self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
		self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
		with tf.variable_scope('eval_net'):
			# c_names(collections_names) are the collections to store variables
			c_names, n_l1, w_initializer, b_initializer = \
				['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
				tf.random_normal_initializer(0., 0.5), tf.constant_initializer(0.1)  # config of layers

			# first layer. collections is used later when assign to target net
			with tf.variable_scope('l1'):
				w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
				b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
				l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

			# second layer. collections is used later when assign to target net
			with tf.variable_scope('l2'):
				w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
				b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
				self.q_eval = tf.matmul(l1, w2) + b2

		with tf.variable_scope('loss'):
			self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
		with tf.variable_scope('train'):
			self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
			#self._train_op = tf.train.AdagradOptimizer(self.lr).minimize(self.loss)

		# ------------------ build target_net ------------------
		self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
		with tf.variable_scope('target_net'):
			# c_names(collections_names) are the collections to store variables
			c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

			# first layer. collections is used later when assign to target net
			with tf.variable_scope('l1'):
				w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
				b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
				l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

			# second layer. collections is used later when assign to target net
			with tf.variable_scope('l2'):
				w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
				b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
				self.q_next = tf.matmul(l1, w2) + b2

	def store_transition(self, s, a, r, s_):
		if not hasattr(self, 'memory_counter'):
			self.memory_counter = 0

		transition = np.hstack((s, [a, r], s_))

		# replace the old memory with new memory
		index = self.memory_counter % self.memory_size

		self.memory[index, :] = transition

		self.memory_counter += 1

	def choose_action(self, observation):
		# to have batch dimension when feed into tf placeholder
		observation = observation[np.newaxis, :]

		if np.random.uniform() <= self.epsilon:
			# forward feed the observation and get q value for every actions
			actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
			action = np.argmax(actions_value)
		else:
			action = np.random.randint(0, self.n_actions)
		return action

	def learn(self):
		# check to replace target parameters
		if self.learn_step_counter % self.replace_target_iter == 0:
			self.sess.run(self.replace_target_op)
			#print('\ntarget_params_replaced\n')

		# sample batch memory from all memory
		if self.memory_counter > self.memory_size:
			sample_index = np.random.choice(self.memory_size, size=self.batch_size)
		else:
			sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
		batch_memory = self.memory[sample_index, :]

		q_next, q_eval = self.sess.run(
			[self.q_next, self.q_eval],
			feed_dict={
				self.s_: batch_memory[:, -self.n_features:],  # fixed params
				self.s: batch_memory[:, :self.n_features],  # newest params
			})

		# change q_target w.r.t q_eval's action
		q_target = q_eval.copy()

		batch_index = np.arange(self.batch_size, dtype=np.int32)
		eval_act_index = batch_memory[:, self.n_features].astype(int)
		reward = batch_memory[:, self.n_features + 1]

		q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

		# train eval network
		_, self.cost = self.sess.run([self._train_op, self.loss],
									 feed_dict={self.s: batch_memory[:, :self.n_features],
												self.q_target: q_target})
		self.cost_his.append(self.cost)

		# increasing epsilon
		self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
		self.learn_step_counter += 1
		return self.cost

	def plot_cost(self):
		plt.plot(np.arange(len(self.cost_his)), self.cost_his)
		plt.ylabel('Cost')
		plt.xlabel('training steps')
		plt.show()

# straight from the TF example, modified to work with 7 classes istead of 10
def cnn_model_fn(features, labels, mode):
	input_layer = tf.reshape(features["x"], [-1, 200, 200, 1])
	conv1 = tf.layers.conv2d(inputs=input_layer,
							filters=32,
							kernel_size=[5, 5],
							padding="same",
							activation=tf.nn.relu)

	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

	conv2 = tf.layers.conv2d(inputs=pool1,
							filters=64,
							kernel_size=[5, 5],
							padding="same",
							activation=tf.nn.relu)
	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

	pool2_flat = tf.reshape(pool2, [-1, 50 * 50 * 64])

	dense = tf.layers.dense(inputs=pool2_flat, units=128, activation=tf.nn.relu)

	dropout = tf.layers.dropout(inputs=dense, rate=0.4, 
								training=mode == tf.estimator.ModeKeys.TRAIN)

	logits = tf.layers.dense(inputs=dropout, units=6)

	predictions = {
		# Generate predictions (for PREDICT and EVAL mode)
		"classes": tf.argmax(input=logits, axis=1),
		"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	}

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
	
	onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=6)

	loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	eval_metric_ops = { "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
	
	return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(description='TouchNet actor-critic example')
	parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
	parser.add_argument('--epsilon', type=float, default=0.6, metavar='G', help='epsilon value for random action (default: 0.6)')
	parser.add_argument('--seed', type=int, default=42, metavar='N', help='random seed (default: 42)')
	parser.add_argument('--batch_size', type=int, default=42, metavar='N', help='batch size (default: 42)')
	parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='interval between training status logs (default: 10)') 
	parser.add_argument('--render', action='store_true', help='render the environment')
	parser.add_argument('--gpu', action='store_true', help='use GPU')
	parser.add_argument('--log', type=str, help='log experiment to tensorboard')
	parser.add_argument('--model_path', type=str, help='path to store/retrieve model at')
	parser.add_argument('--mode', type=str, default="train", help='train/test/all model')
	parser.add_argument('--data_path', type=str, default="../../touchable_data/objects/")
	
	args = parser.parse_args()

	env = SenseEnv(vars(args))

	num_games = 20
	game_length = 1000
	e_greedy_inc = 0.05/game_length # want to increase by 0.05 per game so we spend enough time exploring
	mem_size = num_games*game_length 

	cnn_features_TD = np.zeros((num_games,40000), dtype=np.int8)

	cnn_labels_TD = np.zeros(num_games, dtype=np.int8)
	
	cnn_features_ED = np.zeros((num_games,40000), dtype=np.int8)

	cnn_labels_ED = np.zeros(num_games, dtype=np.int8)

	TD_cnt = 0 # counter to keep track of how many times we touch in the training phase

	RL = DeepQNetwork(n_actions=env.action_space_n(),
					 n_features=env.observation_space(),
					 learning_rate=0.1, e_greedy=0.9,
					 replace_target_iter=100, memory_size=mem_size,
					 e_greedy_increment=e_greedy_inc)   

	if args.mode == "train" or args.mode == "all":
		
		games_where_touched = 0
		total_steps = 0
		
		ep_r = np.zeros(num_games)
		ep_touch = np.zeros(num_games)
		
		for i_episode in range(num_games):
			
			observation = env.reset()
			observation.astype(int, copy=False)# save some memory since it defaults to 64bit
			# overkill for an array of integers

			done = False    
			while not done:

				action = RL.choose_action(observation)
				observation_, reward, done, info = env.step(action)
				
				observation_.astype(int, copy=False)

				# something to consider - should we modify the reward if it's the terminal state and
				# we haven't touched yet? Massive penalty for finishing the round with no touch
				RL.store_transition(observation, action, reward, observation_)      
				
				ep_r[i_episode] += reward
				
				if total_steps > 1000:      
					cost = RL.learn()

				if env.is_touching():
					print('touching at step', env.steps, 'total reward is ', ep_r[i_episode])
					games_where_touched += 1
					cnn_features_TD[TD_cnt] = observation_
					cnn_labels_TD[TD_cnt] = env.class_label
					TD_cnt += 1
					ep_touch[i_episode] += 1
									
				if (env.steps % 500 == 0):
					print('\nepisode: ', i_episode+1, 'step: ', env.steps, 'episode reward ', ep_r[i_episode])            

				observation = observation_
				total_steps += 1
						
			print('episode: ', i_episode+1,
					'ep_r: ', round(ep_r[i_episode], 2),
					' epsilon: ', round(RL.epsilon, 3),
					'alpha', RL.lr)
			print('---------------')

		print('out of ', num_games, 'games we touched ', games_where_touched, 'times')
		plt.plot(range(num_games), ep_r)
		plt.show()
	if args.mode == "test" or args.mode == "all":
		# have to recast the data into np.float32
		cnn_features_TD = np.float32(cnn_features_TD)
		cnn_labels_TD = np.int8(cnn_labels_TD) # no need for an int to be 32 bit
		games_where_touched = 0
		
		print('\n\n\ngetting ready to train classifier\n\n\n')
		classifier = tf.estimator.Estimator(model_fn=cnn_model_fn)
		train_input_fn = tf.estimator.inputs.numpy_input_fn(
			x={"x": cnn_features_TD[:TD_cnt-1]},
			y=cnn_labels_TD[:TD_cnt-1],
			batch_size=num_games,
			num_epochs=None,
			shuffle=True)
		classifier.train(input_fn=train_input_fn,
			steps=2000,
			hooks=None)
		print('\nclassifier trained\n')
		ED_cnt = 0 # counter to keep track of how many times we touch in the evaluation phase

		for i_episode in range(num_games): 
			done = False
			print("\n\nepisode: ", i_episode, "testing classification on a new object\n\n")
			observation = env.reset()
			while not done:     
				action = RL.choose_action(observation)
				observation_, reward, done, info = env.step(action)
				observation = observation_
				if done and env.is_touching():          
					print('\n\ntouching object and storing data to evaluate at end\n\n')          
					
					cnn_features_ED[ED_cnt] = observation_
					cnn_labels_ED[ED_cnt] = env.class_label
					ED_cnt += 1
				elif done:
					print('\n\nfinished episode with no touch. Gonna take a guess\n\n')
					cnn_features_ED[ED_cnt] = observation_
					cnn_labels_ED[ED_cnt] = env.class_label
					ED_cnt += 1
					
		cnn_features_ED = np.float32(cnn_features_ED)
		cnn_labels_ED = np.int8(cnn_labels_ED)

		eval_input_fn = tf.estimator.inputs.numpy_input_fn(
			x={"x": cnn_features_ED[:ED_cnt-1]},
			y=cnn_labels_ED[:ED_cnt-1],
			num_epochs=1,
			shuffle=False)      
		eval_results = classifier.evaluate(input_fn=eval_input_fn)
		print('\n--------------------\n')
		print('These are the results of my evaluations')
		print(eval_results)
		print('\n--------------------\n')
		print('These are the results of my predictions:') 
		pred_input_fn = tf.estimator.inputs.numpy_input_fn(
						x={"x": cnn_features_ED[:ED_cnt-1]},
						num_epochs=1,
						shuffle=False)  
		pred_results = list(classifier.predict(input_fn=pred_input_fn))
		predicted_classes = [p["classes"] for p in pred_results]
		for i, classes in enumerate(predicted_classes):
			print('I predicted ', classes, 'actual class is ', cnn_labels_ED[i])
		print('\n--------------------\n')
		print('In the training phase, I touched the following objects:')
		print(cnn_labels_TD[:TD_cnt])