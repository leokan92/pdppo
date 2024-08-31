import sys
from contextlib import closing

import numpy as np
from io import StringIO

from gym import utils
from gym.envs.toy_text import discrete

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "4x4": ["SFFF", "FHFH", "FFFH", "HFFG"],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
}


def generate_random_map(size=8, p=0.8):
    """Generates a random valid map (one that has a path from start to goal)
    :param size: size of each side of the grid
    :param p: probability that a tile is frozen
    """
    valid = False

    # DFS to check that it's a valid path.
    def is_valid(res):
        frontier, discovered = [], set()
        frontier.append((0, 0))
        while frontier:
            r, c = frontier.pop()
            if not (r, c) in discovered:
                discovered.add((r, c))
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                for x, y in directions:
                    r_new = r + x
                    c_new = c + y
                    if r_new < 0 or r_new >= size or c_new < 0 or c_new >= size:
                        continue
                    if res[r_new][c_new] == "G":
                        return True
                    if res[r_new][c_new] != "H":
                        frontier.append((r_new, c_new))
        return False

    while not valid:
        p = min(1, p)
        res = np.random.choice(["F", "H"], (size, size), p=[p, 1 - p])
        res[0][0] = "S"
        res[-1][-1] = "G"
        valid = is_valid(res)
    return ["".join(x) for x in res]


class FrozenLakeEnv(discrete.DiscreteEnv):
    """
    Winter is here. You and your friends were tossing around a frisbee at the
    park when you made a wild throw that left the frisbee out in the middle of
    the lake. The water is mostly frozen, but there are a few holes where the
    ice has melted. If you step into one of those holes, you'll fall into the
    freezing water. At this time, there's an international frisbee shortage, so
    it's absolutely imperative that you navigate across the lake and retrieve
    the disc. However, the ice is slippery, so you won't always move in the
    direction you intend.
    The surface is described using a grid like the following

        SFFF
        FHFH
        FFFH
        HFFG

    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    G : goal, where the frisbee is located

    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, and zero otherwise.
    """

    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, desc=None, map_name="4x4", is_slippery=True):
        if desc is None and map_name is None:
            desc = generate_random_map()
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype="c")
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)

        nA = 4
        nS = nrow * ncol

        isd = np.array(desc == b"S").astype("float64").ravel()
        isd /= isd.sum()

        P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)
        
        goal_position = None
        for row in range(nrow):
            for col in range(ncol):
                if desc[row, col] == b'G':
                    goal_position = (row, col)
                    break
            if goal_position:
                break
        self.goal_position = goal_position

        def proximity_reward(current_row, current_col):
            goal_row, goal_col = goal_position
            distance = abs(goal_row - current_row) + abs(goal_col - current_col)
            return 1.0 / (1.0 + distance)

        def update_probability_matrix(row, col, a):
            newrow, newcol = inc(row, col, a)
            newstate = to_s(newrow, newcol)
            newletter = desc[newrow, newcol]
            terminated = bytes(newletter) in b"G"
            reward = float(newletter == b"G")
            if not terminated:
                reward = proximity_reward(newrow, newcol) + float(newletter == b"H")* -(1/(nrow+ncol))
            return newstate, reward, terminated

        np.random.seed(42)  # Set a seed for reproducibility
        tile_probabilities = np.random.dirichlet(np.ones(4), size=(nrow, ncol))

        def to_row_col(s):
            return divmod(s, ncol)

        base_slip_prob= 0.50
        
        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in b"GH":
                        li.append((1.0, s, 0, True))
                    else:
                        if is_slippery:
                            # First, the agent moves in the desired direction
                            newstate, reward, terminated = update_probability_matrix(row, col, a)
                            if terminated:
                                li.append((1.0, newstate, reward, terminated))
                            else:
                                # After the first move, slippery condition causes an additional movement
                                row2, col2 = to_row_col(newstate)
                                for b, prob in enumerate(tile_probabilities[row2, col2]):
                                    newstate_post, reward_pos, terminated_post = update_probability_matrix(row2, col2, b)
                                    li.append(
                                        (base_slip_prob * prob, newstate_post, reward_pos + reward, terminated_post)
                                    )
                                # Add the remaining probability for staying at the newstate
                                li.append((1.0 - base_slip_prob, newstate, reward, False))
                        else:
                            li.append((1.0, *update_probability_matrix(row, col, a)))

        self.P = P
        super(FrozenLakeEnv, self).__init__(nS, nA, P, isd)

    def get_post_decision_state(self, s, a):
        def proximity_reward(current_row, current_col):
            goal_row, goal_col = self.goal_position
            distance = abs(goal_row - current_row) + abs(goal_col - current_col)
            return 1.0 / (1.0 + distance)

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, self.nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, self.ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)

        def to_s(row, col):
            return row * self.ncol + col

        def to_row_col(s):
            row = s // self.ncol
            col = s % self.ncol
            return row, col

        row, col = to_row_col(s)
        next_row, next_col = inc(row, col, a)
        next_s = to_s(next_row, next_col)
        next_r = proximity_reward(next_row, next_col)
        return next_s, next_r
    
    def render(self, mode="human"):
        outfile = StringIO() if mode == "ansi" else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode("utf-8") for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write(
                "  ({})\n".format(["Left", "Down", "Right", "Up"][self.lastaction])
            )
        else:
            outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in desc) + "\n")

        if mode != "human":
            with closing(outfile):
                return outfile.getvalue()
