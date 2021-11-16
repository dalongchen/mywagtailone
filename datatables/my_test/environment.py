"""
强化学习
"""
from enum import Enum
import numpy as np


class State():

    def __init__(self, row=-1, column=-1):
        self.row = row
        self.column = column

    def __repr__(self):
        return "<State: [{}, {}]>".format(self.row, self.column)

    def clone(self):
        return State(self.row, self.column)

    def __hash__(self):
        return hash((self.row, self.column))

    def __eq__(self, other):
        return self.row == other.row and self.column == other.column


class Action(Enum):
    UP = 1
    DOWN = -1
    LEFT = 2
    RIGHT = -2


class Environment():

    def __init__(self, grid, move_prob=0.8):
        # grid是一个二维数组，它的值可以看作属性
        # 一些属性的情况如下
        #  0: 普通格子
        #  -1: 有危险的格子 (游戏结束)
        #  1: 有奖励的格子 (游戏结束)
        #  9: 被屏蔽的格子 (无法放置智能体)
        self.grid = grid
        self.agent_state = State()

        # 默认的奖励是负数，就像施加了初始位置惩罚
        # 这意味着智能体必须快速到达终点
        self.default_reward = -0.04

        # 智能体能够以 move_prob 的概率向所选方向移动
        # 如果概率值在(1 - move_prob)内
        # 则意味着智能体将移动到不同的方向
        self.move_prob = move_prob
        self.reset()

    @property
    def row_length(self):
        return len(self.grid)

    @property
    def column_length(self):
        return len(self.grid[0])

    @property
    def actions(self):
        return [Action.UP, Action.DOWN,
                Action.LEFT, Action.RIGHT]

    @property
    def states(self):
        states = []
        for row in range(self.row_length):
            for column in range(self.column_length):
                # state中不包含被屏蔽的格子
                if self.grid[row][column] != 9:
                    states.append(State(row, column))
        return states

    # state开始 = [2, 0]，action随机的上下左右action= action Action.RIGHT,Action.DOWN, Action.UP
    def transit_func(self, state, action):
        transition_probs = {}
        if not self.can_action_at(state):
            # 已经到达游戏结束的格子
            return transition_probs
        # print("action.value", action.value)
        # print("action.value", action.value * -1)
        # opposite_direction =Action.LEFT,right,up,down和action相反
        opposite_direction = Action(action.value * -1)
        # print("opposite_direction", opposite_direction)
        # self.actions= [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
        for a in self.actions:
            # print("action", action)
            # print("a", a)
            # prob=0, 0.8, 0.09999
            # 第一回，agent向up动，prob=0.8，向down，为0，向left，right为0.1
            prob = 0
            if a == action:
                # self.move_prob=move_prob = 0.8
                prob = self.move_prob
            elif a != opposite_direction:
                prob = (1 - self.move_prob) / 2
            print("prob", prob)
            next_state = self._move(state, a)
            if next_state not in transition_probs:
                transition_probs[next_state] = prob
            else:
                transition_probs[next_state] += prob
        print("transition_probs", transition_probs)
        return transition_probs

    def can_action_at(self, state):
        if self.grid[state.row][state.column] == 0:
            return True
        else:
            return False

    def _move(self, state, action):
        if not self.can_action_at(state):
            raise Exception("Can't move from here!")

        next_state = state.clone()

        # 执行行动（移动）
        if action == Action.UP:
            next_state.row -= 1
        elif action == Action.DOWN:
            next_state.row += 1
        elif action == Action.LEFT:
            next_state.column -= 1
        elif action == Action.RIGHT:
            next_state.column += 1

        # 检查状态是否在grid外
        if not (0 <= next_state.row < self.row_length):
            next_state = state
        if not (0 <= next_state.column < self.column_length):
            next_state = state

        # 检查智能体是否到达了被屏蔽的格子
        if self.grid[next_state.row][next_state.column] == 9:
            next_state = state

        return next_state

    def reward_func(self, state):
        reward = self.default_reward  # self.default_reward = -0.04
        done = False

        # 检查下一种状态的属性
        attribute = self.grid[state.row][state.column]
        if attribute == 1:
            # 获取奖励，游戏结束
            reward = 1
            done = True
        elif attribute == -1:
            # 遇到危险，游戏结束
            reward = -1
            done = True

        return reward, done

    def reset(self):
        # 将智能体放置到左下角
        self.agent_state = State(self.row_length - 1, 0)
        return self.agent_state

    # action随机的上下左右action随机的上下左右action= action Action.RIGHT,Action.DOWN, Action.UP
    def step(self, action):
        next_state, reward, done = self.transit(self.agent_state, action)
        if next_state is not None:
            self.agent_state = next_state

        return next_state, reward, done

    # 迁移函数  state开始=[2,0], action随机的上下左右action随机的上下左右action= action Action.RIGHT,Action.DOWN, Action.UP
    def transit(self, state, action):
        transition_probs = self.transit_func(state, action)
        if len(transition_probs) == 0:
            return None, None, True

        next_states = []
        probs = []
        for s in transition_probs:
            next_states.append(s)
            probs.append(transition_probs[s])

        next_state = np.random.choice(next_states, p=probs)
        reward, done = self.reward_func(next_state)
        return next_state, reward, done
