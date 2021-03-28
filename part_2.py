import os
from typing import Any, Iterable, Tuple
import random

# inital setup
os.makedirs("outputs", exist_ok=True)

# Consts
TEAM_NO = 93
_Y = [1/2, 1, 2][TEAM_NO % 3]
STEP_COST = -10/_Y
GAMMA = 0.999
DELTA = 1e-3


def prob(p: float):
    """Returns True with a probability if p"""
    return random.random() < p


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    # https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary#comment77092107_23689767
    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    # https://stackoverflow.com/a/32453194/8608146

    def __dir__(self) -> Iterable[str]:
        l = list(dict.keys(self))
        l.extend(super(DotDict, self).__dir__())
        return l


pl_actions = DotDict({"LEFT": (-1, 0), "RIGHT": (1, 0),
                      "UP": (0, 1), "DOWN": (0, -1), "STAY": (0, 0)})
pl_states = DotDict({"W": (-1, 0), "E": (1, 0),
                     "N": (0, 1), "S": (0, -1), "C": (0, 0)})
pl_inv_states = {pl_states[k]: k for k in pl_states}


class Player(object):
    ACTIONS = pl_actions
    STATES = pl_states
    _INV_STATE_MAP = pl_inv_states

    def __init__(self, name: str, inital_state: Tuple = STATES.N):
        self.name = name
        self.state = inital_state

    def move(self, direction: tuple):
        old_pos = self.state
        # add direction to state
        # state += direction
        self.state = tuple(map(sum, zip(self.state, direction)))
        if (sum([abs(x) for x in self.state])) > 1:
            # out of bounds undo move
            self.state = old_pos

    @property
    def cur_state(self) -> str:
        return Player._INV_STATE_MAP[self.state]

    def jump_to_east(self):
        self.state = Player.STATES.E

    def prob_move(self, direction: tuple):
        if self.state in [Player.STATES.C, Player.STATES.N, Player.STATES.S]:
            if not prob(0.85):
                # move/teleport to (E) 15% of time
                self.jump_to_east()
                return
        # any other move for any state is determined i.e. prob = 1
        self.move(direction)

    def __str__(self):
        return self.name


class Enemy(object):
    STATE = DotDict({"DORMANT": 0, "READY": 1})

    def __init__(self, name: str):
        self.name = name
        self.health = 100
        self.state = Enemy.STATE.DORMANT

    def __str__(self):
        return self.name


def loop():
    ij = Player("Indiana Jones")
    mm = Enemy("Mighty Monster")
    print(ij, "vs", mm)
    print("Start!")
    max_tries = 100
    iter_count = 0
    while True:
        iter_count += 1
        # print("Round", iter_count)
        ij.prob_move(random.choice(list(Player.ACTIONS.values())))
        print(ij.cur_state)
        if iter_count >= max_tries:
            print("Game Over")
            return


def main():
    print("Params", "DELTA", DELTA, "GAMMA", GAMMA, "STEP_COST", STEP_COST)
    loop()


if __name__ == '__main__':
    main()
