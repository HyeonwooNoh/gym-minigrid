import numpy as np
import itertools

from gym_minigrid.minigrid import *
from gym_minigrid.register import register

RNG_STATE = np.random.RandomState(123)

ACTION_SEMANTICS_ALL = list(set(
    [i[:3] for i in itertools.permutations([0, 1, 2, 3, 4, 5, 6])]))
sorted(ACTION_SEMANTICS_ALL)
ACTION_SEMANTICS_ALL_INDEX = list(range(len(ACTION_SEMANTICS_ALL)))
RNG_STATE.shuffle(ACTION_SEMANTICS_ALL_INDEX)
ACTION_SEMANTICS_ALL_INDEX_TRAIN = ACTION_SEMANTICS_ALL_INDEX[:150]
ACTION_SEMANTICS_ALL_INDEX_TEST = ACTION_SEMANTICS_ALL_INDEX[150:]


class Actions():
    def __init__(self, action_order=[0,1,2]):
        self.left = action_order[0]
        self.right = action_order[1]
        self.forward = action_order[2]
        action_order_set = set(action_order)
        remaining = [i for i in range(7) if i not in action_order_set]
        self.pickup = remaining[0]
        self.drop = remaining[1]
        self.toggle = remaining[2]
        self.done = remaining[3]


class RandomDynamicsEnv(MiniGridEnv):
    """
    Empty grid, random dynamics, no obstacles, sparse reward
    """

    def __init__(
        self,
        size=8,
        agent_start_pos=(1,1),
        agent_start_dir=0,
        train=True,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        if train:
            self.action_table_index = ACTION_SEMANTICS_ALL_INDEX_TRAIN
        else:
            self.action_table_index = ACTION_SEMANTICS_ALL_INDEX_TEST
        self._i = 0

        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _sample_action_semantic(self):
        action_table = ACTION_SEMANTICS_ALL[self.action_table_index[self._i]]
        self.actions=Actions(action_table)
        self._i += 1
        if self._i > len(self.action_table_index):
            np.random.shuffle(self.action_table_index)
            self._i = 0

    def _gen_grid(self, width, height):
        # Sample action semantic (random dynamics)
        self._sample_action_semantic()

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.grid.set(width - 2, height - 2, Goal())

        # Place the agent
        if self.agent_start_pos is not None:
            self.start_pos = self.agent_start_pos
            self.start_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"


class RandomDynamicsEnv5x5(RandomDynamicsEnv):
    def __init__(self):
        super().__init__(size=5, agent_start_pos=None)

class RandomDynamicsEnv7x7Train(RandomDynamicsEnv):
    def __init__(self):
        super().__init__(size=7, agent_start_pos=None, train=True)

class RandomDynamicsEnv7x7Test(RandomDynamicsEnv):
    def __init__(self):
        super().__init__(size=7, agent_start_pos=None, train=False)

register(
    id='MiniGrid-RandomDynamics-5x5-v0',
    entry_point='gym_minigrid.envs:RandomDynamicsEnv5x5'
)
register(
    id='MiniGrid-RandomDynamics-7x7-train-v0',
    entry_point='gym_minigrid.envs:RandomDynamicsEnv7x7Train'
)
register(
    id='MiniGrid-RandomDynamics-7x7-test-v0',
    entry_point='gym_minigrid.envs:RandomDynamicsEnv7x7Test'
)
