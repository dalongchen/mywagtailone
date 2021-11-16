import random
from mywagtailone.datatables.my_test.environment import Environment


class Agent():
    # env.actions= [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
    def __init__(self, env):
        self.actions = env.actions

    def policy(self, state):
        return random.choice(self.actions)


def main():
    # 生成grid环境
    grid = [
        [0, 0, 0, 1],
        [0, 9, 0, -1],
        [0, 0, 0, 0]
    ]
    env = Environment(grid)
    agent = Agent(env)

    # 尝试10次游戏
    for i in range(1):
        # 初始化智能体的位置
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # state,没有用？action随机的上下左右action= action Action.RIGHT,Action.DOWN, Action.UP
            action = agent.policy(state)
            # print("action", action)
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state

        print("Episode {}: Agent gets {} reward.".format(i, total_reward))


if __name__ == "__main__":
    main()
