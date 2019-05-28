#!/usr/bin/env python3

from __future__ import division, print_function

import sys
import numpy
import gym
import time
from optparse import OptionParser

import gym_minigrid

def main():
    parser = OptionParser()
    parser.add_option(
        "-e",
        "--env-name",
        dest="env_name",
        help="gym environment to load",
        default='MiniGrid-MultiRoom-N6-v0'
    )
    (options, args) = parser.parse_args()

    # Load the gym environment
    env = gym.make(options.env_name)

    def resetEnv():
        env.reset()
        if hasattr(env, 'mission'):
            print('Mission: %s' % env.mission)

    resetEnv()

    # Create a window to render into
    renderer = env.render('human')

    def keyDownCb(keyName):
        if keyName == 'BACKSPACE':
            resetEnv()
            return

        if keyName == 'ESCAPE':
            sys.exit(0)

        action = 0
        print(keyName)
        if keyName == 'LEFT' or keyName == '0':
            action = env.std_actions.left
        elif keyName == 'RIGHT' or keyName == '1':
            action = env.std_actions.right
        elif keyName == 'UP' or keyName == '2':
            action = env.std_actions.forward

        elif keyName == 'SPACE' or keyName == '3':
            action = env.std_actions.toggle
        elif keyName == 'PAGE_UP' or keyName == '4':
            action = env.std_actions.pickup
        elif keyName == 'PAGE_DOWN' or keyName == '5':
            action = env.std_actions.drop

        elif keyName == 'RETURN' or keyName == '6':
            action = env.std_actions.done

        else:
            print("unknown key %s" % keyName)
            return

        obs, reward, done, info = env.step(action)

        print('step=%s, reward=%.2f' % (env.step_count, reward))

        if done:
            print('done!')
            resetEnv()

    renderer.window.setKeyDownCb(keyDownCb)

    while True:
        env.render('human')
        time.sleep(0.01)

        # If the window was closed
        if renderer.window == None:
            break

if __name__ == "__main__":
    main()
