#!/usr/bin/env python
import argparse
import logging
import random
import sys
import sensenet
#import gym

logger = logging.getLogger()

def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    parser.add_argument('-e', '--environment', dest='environment', default='HandEnv-v0', help='Set environment')
    args = parser.parse_args()

    if args.verbosity == 0:
        logger.setLevel(logging.INFO)
    elif args.verbosity >= 1:
        logger.setLevel(logging.DEBUG)

    env = sensenet.make(args.environment,{'render':True})
    #env = gym.make('MountainCar-v0')
    
    observation_n = env.reset()

    while True:
        # your agent here
        env.render()
        action_n = env.action_space.sample()
        observation_n, reward_n, done_n, info = env.step(action_n)

    return 0

if __name__ == '__main__':
    sys.exit(main())
