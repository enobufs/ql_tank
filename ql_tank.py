import random as rand 
import math
import readchar
import argparse
import matplotlib.pyplot as plt


ALPHA = 0.5     # learning rate
NOISE = 0.2     # known as epsilon
GAMMA = 0.9     # discount rate
TCOST = -0.04   # cost for transition
EPSILON = 0.1   # choose random action at this ratio

CELL_ROOM = 0 # can enter
CELL_WALL = 1 # cannot enter
CELL_SUCC = 2 # success and exit
CELL_FAIL = 3 # fail and exit


class Environment:

    def __init__(self):
        self.shape = (3, 4) # (height, width)
        self.cur_pos = (0, 0)
        self.map = [
            [CELL_ROOM, CELL_ROOM, CELL_ROOM, CELL_ROOM],
            [CELL_ROOM, CELL_WALL, CELL_ROOM, CELL_FAIL],
            [CELL_ROOM, CELL_ROOM, CELL_ROOM, CELL_SUCC]
        ]
        self.reward = [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 0.0, 1.0]
        ]

    def action(self, action_name):
        pos = self.cur_pos
        if pos == (3, 2) or pos == (3, 1):
            raise Exception("No action is allowed in the end state.")

        next_x = self.cur_pos[0]
        next_y = self.cur_pos[1]

        if action_name == "u":
            if self.cur_pos[1] != self.shape[0] - 1:
                next_y += 1
        elif action_name == "d":
            if self.cur_pos[1] != 0:
                next_y -= 1
        elif action_name == "l":
            if self.cur_pos[0] != 0:
                next_x -= 1
        elif action_name == "r":
            if self.cur_pos[0] != self.shape[1] - 1:
                next_x += 1
        else:
            raise Exception("Invalid action: %s" % (action_name)) 

        reward = self.reward[next_y][next_x]

        # Check cell type
        ct = self.map[next_y][next_x]
        if ct != CELL_SUCC and ct != CELL_FAIL:
            if ct == CELL_WALL:
                # Move back to the original location
                next_x = self.cur_pos[0]
                next_y = self.cur_pos[1]

            reward = TCOST

        # Update the positions
        self.pre_pos = self.cur_pos
        self.cur_pos = (next_x, next_y)

        return self.cur_pos, reward

    def reset(self):
        self.pre_pos = None
        self.cur_pos = (0, 0)
        return self.cur_pos


def max_a(s):
    return max(s["u"], s["d"], s["r"], s["l"])

def to_state_name(state):
    return "%d-%d" % state

def new_state(name, pos, ct):
    return {
        "name": name,   # name of this state
        "pos": pos,     # (x, y)
        "ct": ct,       # cell type for this state
        "u": 0.0,       # utility for action, 'up'
        "d": 0.0,       # utility for action, 'down'
        "l": 0.0,       # utility for action, 'left'
        "r": 0.0        # utility for action, 'right'
    }

def by_utility(s):
    return s["utility"]


class Agent:
    def __init__(self):
        self.auto = True
        self.episode = 0
        self.move = 0
        self.errors = 0
        self.error_rates = []
        self.env = Environment()
        self.reset()
        self.S = { self.cur_st_name: new_state(self.cur_st_name, (0, 0), CELL_ROOM) }
        self.draw()

    def train(self, auto, episodes):
        self.auto = auto

        while True:
            if self.auto:

                if self.episode == episodes:
                    break

                cur_st = self.S[self.cur_st_name]
                actions = [
                        { "act": "u", "utility": cur_st["u"] },
                        { "act": "d", "utility": cur_st["d"] },
                        { "act": "l", "utility": cur_st["l"] },
                        { "act": "r", "utility": cur_st["r"] }
                ]
                actions = sorted(actions, key=by_utility, reverse=True)

                idx = 0
                msg = "BEST"
                if rand.random() < EPSILON:
                    # choose randomly from index [1, 3]
                    idx = math.floor(rand.random() * 3) + 1
                    msg = "RAND"
                act = actions[idx]["act"];
                print("Action chosen: %s (%s)" % (act, msg))

            else:
                key = readchar.readkey()
                if key == '\x1b[A':
                    act = "u"
                elif key == '\x1b[B':
                    act = "d"
                elif key == '\x1b[C':
                    act = "r"
                elif key == '\x1b[D':
                    act = "l"
                elif key == '\r':
                    break
                else:
                    print("Aborted")
                    continue

            self.move += 1

            s, r = self.env.action(act)
            new_st_name = to_state_name(s)
            ct = self.env.map[s[1]][s[0]]
            if not new_st_name in self.S:
                new_st = new_state(new_st_name, s, ct)
                self.S[new_st_name] = new_st
            else:
                new_st = self.S[new_st_name]

            # Update the current state name
            self.pre_st_name = self.cur_st_name
            self.cur_st_name = new_st_name
            self.last_action = act

            self._update(r)

            if math.fabs(r) == 1.0:
                if r < 0.0:
                    self.errors += 1
                self.reset()

            self.draw()

        if self.auto:
            plt.plot(self.error_rates)
            plt.ylabel("Error ratio")
            plt.xlabel("Episodes")
            plt.show()

    def _update(self, r):
        pre_st = self.S[self.pre_st_name]
        cur_st = self.S[self.cur_st_name]

        pre_st[self.last_action] += ALPHA * (r + GAMMA * max_a(cur_st) - pre_st[self.last_action])

    def reset(self):
        self.episode += 1
        self.move = 0
        self.cur_st_name = to_state_name(self.env.reset())
        self.pre_st_name = None
        self.last_action = None
        if self.episode % 5 == 0:
            self.error_rates.append(self.errors / self.episode)

    def draw(self):
        pos = self.env.cur_pos
        print("EPISODE: %d, MOVE: %d, ERROR RATE: %.2f" % (self.episode, self.move, self.errors/self.episode))
        print("+-------------+-------------+-------------+-------------+")
        for ry in range(self.env.shape[0]):
            y = self.env.shape[0] - 1 - ry
            l1 = "|"
            l2 = "|"
            l3 = "|"
            for x in range(self.env.shape[1]):
                sn = to_state_name((x, y))
                s = None
                if sn in self.S:
                    s = self.S[sn]

                ct = self.env.map[y][x]
                if ct == CELL_WALL:
                    l1 += "#############|"
                    l2 += "#############|"
                    l3 += "#############|"
    
                else:
                    if s == None:
                        # Unknown location
                        l1 += "             |"
                        l2 += "      ?      |"
                        l3 += "             |"
                    else:
                        marker = " "
                        if (x, y) == self.env.cur_pos:
                            marker = "*"
                        elif (x, y) == self.env.pre_pos:
                            marker = "+"

                        if s["ct"] == CELL_SUCC:
                            l1 += "+-----------+|"
                            l2 += "|  Success  ||"
                            l3 += "+-----------+|"
                        elif s["ct"] == CELL_FAIL:
                            l1 += "+-----------+|"
                            l2 += "|    Fail   ||"
                            l3 += "+-----------+|"
                        else:
                            l1 += "    %+.2f    |" % (s["u"])
                            l2 += "%+.2f %s %+.2f|" % (s["l"], marker, s["r"])
                            l3 += "    %+.2f    |" % (s["d"])

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
