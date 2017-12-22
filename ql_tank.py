import random as rand
import numpy as np
import math
import argparse
import readchar
import matplotlib.pyplot as plt

# Global constants
ALPHA = 0.5     # learning rate
NOISE = 0.2     # known as epsilon
GAMMA = 0.9     # discount rate
TCOST = -0.04   # cost for transition
EPSILON = 0.1   # choose random action at this ratio

CELL_ROOM = 0 # can enter
CELL_WALL = 1 # cannot enter
CELL_SUCC = 2 # success and exit
CELL_FAIL = 3 # fail and exit

ACTION_UP = 0
ACTION_DOWN = 1
ACTION_RIGHT = 2
ACTION_LEFT = 3

WIDTH = 4  # x
HEIGHT = 3 # y
N_STATES = WIDTH * HEIGHT
N_ACTIONS = 4

def to_action_name(action):
    if action == ACTION_UP:
        return "up"
    if action == ACTION_DOWN:
        return "down"
    if action == ACTION_RIGHT:
        return "right"
    if action == ACTION_LEFT:
        return "left"

    raise Exception("Invalid action: %d" % (action))


class Environment:

    def __init__(self):
        # To get cell type by position.
        # ex) cell type at (x, y): self.map[y][x]
        self.map = [
            [CELL_ROOM, CELL_ROOM, CELL_ROOM, CELL_ROOM],
            [CELL_ROOM, CELL_WALL, CELL_ROOM, CELL_FAIL],
            [CELL_ROOM, CELL_ROOM, CELL_ROOM, CELL_SUCC]
        ]
        # To get reward by position.
        # ex) reward at (x, y): self.reward[y][x]
        self.reward = [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 0.0, 1.0]
        ]
        # To get state by position.
        # ex) state at (x, y): self.states[y][x]
        self.states = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11]
        ]

        self.reset()

    def pos2state(self, pos):
        return self.states[pos[1]][pos[0]]

    def state2pos(self, s):
        return (s % WIDTH, math.floor(s / HEIGHT))

    def cur_state(self):
        return self.pos2state(self.cur_pos)

    def pre_state(self):
        if self.pre_pos == None:
            return None
        return self.pos2state(self.pre_pos)

    def is_ended(self):
        ct = self.map[self.cur_pos[1]][self.cur_pos[0]]
        return ct == CELL_SUCC or ct == CELL_FAIL

    def action(self, action):
        pos = self.cur_pos
        if pos == (3, 2) or pos == (3, 1):
            raise Exception("No action is allowed in the end state.")

        next_x, next_y = self.cur_pos

        if action == ACTION_UP:
            if self.cur_pos[1] != HEIGHT - 1:
                next_y += 1
        elif action == ACTION_DOWN:
            if self.cur_pos[1] != 0:
                next_y -= 1
        elif action == ACTION_LEFT:
            if self.cur_pos[0] != 0:
                next_x -= 1
        elif action == ACTION_RIGHT:
            if self.cur_pos[0] != WIDTH - 1:
                next_x += 1
        else:
            raise Exception("Invalid action: %s" % (to_action_name(action))) 

        reward = self.reward[next_y][next_x]

        # Check cell type
        ct = self.map[next_y][next_x]
        if ct != CELL_SUCC and ct != CELL_FAIL:
            if ct == CELL_WALL:
                # Move back to the original location
                next_x, next_y = self.cur_pos

            reward = TCOST

        # Update the positions
        self.pre_pos = self.cur_pos
        self.cur_pos = (next_x, next_y)

        return self.cur_state(), reward

    def reset(self):
        self.pre_pos = None
        self.cur_pos = (0, 0)
        return self.cur_state()


class Agent:
    def __init__(self):
        self.auto = True
        self.episode = 0
        self.move = 0
        self.errors = 0
        self.error_rates = []
        self.env = Environment()
        self.reset()
        self.Q = np.zeros((N_STATES, N_ACTIONS))
        self.draw()

    def train(self, auto, episodes):
        self.auto = auto

        while True:
            if self.auto:
                # Automatic trainig
                if self.episode == episodes:
                    break

                action = 0
                if rand.random() < EPSILON:
                    # choose the action randomly
                    action = math.floor(rand.random() * 4)
                    msg = "RAND"
                else:
                    # pick the best action based on Q-values for the current state
                    action = np.argmax(self.Q[self.env.cur_state()])
                    msg = "BEST"

                print("Action chosen: %s (%s)" % (to_action_name(action), msg))

            else:
                # Manual trainig
                # Wait for an arrow key input.
                key = readchar.readkey()
                if key == '\x1b[A':
                    action = ACTION_UP
                elif key == '\x1b[B':
                    action = ACTION_DOWN
                elif key == '\x1b[C':
                    action = ACTION_RIGHT
                elif key == '\x1b[D':
                    action = ACTION_LEFT
                else:
                    print("Aborted")
                    break

            self.move += 1

            # Make action to the environment.
            # The environment returns the new state and the immediate reward.
            s, r = self.env.action(action)

            # Update the current state
            self.last_action = action

            # Update the Q table
            self._update_Q(r)

            # If the current state is the end state, go back to the initial state.
            if self.env.is_ended():
                if r < 0.0:
                    self.errors += 1
                self.reset()

            self.draw()

        if self.auto:
            print("Plotting...")
            plt.plot(self.error_rates)
            plt.ylabel("Error ratio")
            plt.xlabel("(x5) Episodes")
            plt.show()
            print("Done")

    def _update_Q(self, r):
        s_pre = self.env.pre_state() # s(t)
        s_cur = self.env.cur_state() # s(t+1)
        a = self.last_action;

        # The heart of Q-Learning
        self.Q[s_pre, a] += ALPHA * (r + GAMMA * np.amax(self.Q[s_cur]) - self.Q[s_pre, a])

    def reset(self):
        self.env.reset()
        self.episode += 1
        self.move = 0
        self.last_action = None
        if self.episode % 5 == 0:
            self.error_rates.append(self.errors / self.episode)

    def draw(self):
        pos = self.env.cur_pos
        print("EPISODE: %d, MOVE: %d, ERROR RATE: %.2f" % (self.episode, self.move, self.errors/self.episode))
        print("+-------------+-------------+-------------+-------------+")
        for ry in range(HEIGHT):
            y = HEIGHT - 1 - ry
            l1 = "|"
            l2 = "|"
            l3 = "|"
            for x in range(WIDTH):
                ct = self.env.map[y][x]
                if ct == CELL_WALL:
                    l1 += "#############|"
                    l2 += "#############|"
                    l3 += "#############|"
                elif ct == CELL_SUCC:
                    l1 += "+-----------+|"
                    l2 += "|  Success  ||"
                    l3 += "+-----------+|"
                elif ct == CELL_FAIL:
                    l1 += "+-----------+|"
                    l2 += "|    Fail   ||"
                    l3 += "+-----------+|"
                else:
                    s = self.env.pos2state((x, y))
                    marker = " "
                    if s == self.env.cur_state():
                        marker = "*"
                    elif s == self.env.pre_state():
                        marker = "+"

                    l1 += "    %+.2f    |" % (self.Q[s, ACTION_UP])
                    l2 += "%+.2f %s %+.2f|" % (self.Q[s, ACTION_LEFT], marker, self.Q[s, ACTION_RIGHT])
                    l3 += "    %+.2f    |" % (self.Q[s, ACTION_DOWN])

            print(l1)
            print(l2)
            print(l3)
            print("+-------------+-------------+-------------+-------------+")

        print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--auto', help='Trains automatically.', action='store_true')
    parser.add_argument('--episodes', help='Number of episodes to run', type=int, default=400)

    args = parser.parse_args()
    print("Auto-training: ", args.auto)
    if args.auto:
        print("Episodes     : ", args.episodes)
    else:
        print(">>> Use arrow keys to perform actions. Hit enter to exit. <<<")

    agent = Agent()
    agent.train(args.auto, args.episodes)
