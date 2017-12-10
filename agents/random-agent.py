#!/usr/bin/env python
import argparse
import logging
import sys

import sys
sys.path.append('..')
import sensenet
import random
from sensenet.envs.handroid.hand_env import HandEnv
logger = logging.getLogger()

def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    args = parser.parse_args()

    if args.verbosity == 0:
        logger.setLevel(logging.INFO)
    elif args.verbosity >= 1:
        logger.setLevel(logging.DEBUG)


    env = HandEnv({'render':True})
    
    observation_n = env.reset()

    while True:
        # your agent here
        #
        # Try sending this instead of a random action: ('KeyEvent', 'ArrowUp', True)
        action_n = random.choice(env.action_space())
        observation_n, reward_n, done_n, info = env.step(action_n)

    return 0

if __name__ == '__main__':
    sys.exit(main())
