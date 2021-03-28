import os
from typing import Iterable, Tuple
import random
import numpy as np

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


class Move:
    LEFT = np.array((-1, 0))
    RIGHT = np.array((1, 0))
    UP = np.array((0, 1))
    DOWN = np.array((0, -1))
    STAY = np.array((0, 0))


class PlayerState:
    W = np.array((-1, 0))
    E = np.array((1, 0))
    N = np.array((0, 1))
    S = np.array((0, -1))
    C = np.array((0, 0))


# unit vectors for directions
pl_moves = DotDict({
    "LEFT": tuple(Move.LEFT),
    "RIGHT": tuple(Move.RIGHT),
    "UP": tuple(Move.UP),
    "DOWN": tuple(Move.DOWN),
    "STAY": tuple(Move.STAY)
})
pl_states = DotDict({
    "W": tuple(PlayerState.W),
    "E": tuple(PlayerState.E),
    "N": tuple(PlayerState.N),
    "S": tuple(PlayerState.S),
    "C": tuple(PlayerState.C)
})

_INV_STATE_MAP = {pl_states[k]: k for k in pl_states}

pl_actions = DotDict(
    {"SHOOT": 25, "HIT": 50, "CRAFT": 2, "GATHER": 3, "NONE": 4})


class Player(object):
    MOVES = pl_moves
    STATES = pl_states
    ATTACKS = pl_actions

    def __init__(self, name: str, inital_state=PlayerState.N):
        self.name = name
        self.state = inital_state
        self.arrows = 3
        self.materials = 0
        self.reward = 0

    def move(self, direction):
        # add direction to state
        self.state += direction
        if np.sum(np.abs(self.state)) > 1:
            # out of bounds, illegal move -> undo move
            self.state -= direction

    @property
    def cur_state(self) -> str:
        return _INV_STATE_MAP[tuple(self.state)]

    def jump_to_east(self):
        self.state = PlayerState.E

    def try_move(self, direction: tuple):
        if tuple(self.state) in [Player.STATES.C, Player.STATES.N, Player.STATES.S]:
            if not prob(0.85):
                # move/teleport to (E) 15% of time
                self.jump_to_east()
                return
        # any other move for any state is determined i.e. prob = 1
        self.move(direction)

    def defend(self, enemy):
        if tuple(self.state) not in [Player.STATES.C, Player.STATES.E]:
            return
        self.arrows = 0
        self.reward -= 40
        enemy.heal()

    def attack(self, enemy, action: int):
        success_prob = 0
        if action == Player.ATTACKS.SHOOT:
            # check arrows
            if self.arrows == 0:
                return
            self.arrows -= 1
            if np.all(self.state, PlayerState.C):
                success_prob = 0.5
            elif np.all(self.state, PlayerState.W):
                success_prob = 0.25
            elif np.all(self.state, PlayerState.E):
                success_prob = 0.9
        elif action == Player.ATTACKS.HIT:
            if np.all(self.state, PlayerState.C):
                success_prob = 0.1
            elif np.all(self.state, PlayerState.E):
                success_prob = 0.2

        # deal appropriate damage with success_probability
        if prob(success_prob):
            enemy.bleed(action)

    def craft(self):
        if self.state != Player.STATES.N or self.materials == 0:
            return
        self.materials -= 1
        self.arrows += np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])

    def gather(self):
        if self.state != Player.STATES.S:
            return
        if prob(0.75):
            if self.materials == 0:
                self.materials += 1

    def perform_action(self, action):
        if action in Player.ATTACKS:
            print(action)

    def __str__(self):
        return self.name


class Enemy(object):
    STATES = DotDict({"D": 0, "R": 1})

    def __init__(self, name: str):
        self.name = name
        self.health = 100
        self.state = Enemy.STATES.D

    def bleed(self, damage: int = 25):
        self.health -= damage

    def heal(self, health: int = 25):
        self.health += health

    @property
    def dead(self):
        return self.health <= 0

    def get_ready(self):
        self.state = Enemy.STATES.R

    def idle(self):
        self.state = Enemy.STATES.D

    def think(self):
        if prob(0.2):
            self.get_ready()

    def try_attack(self):
        if self.state == Enemy.STATES.R:
            if prob(0.5):
                # todo attack
                self.idle()

    def do_your_thing(self):
        self.think()
        self.try_attack()

    def __str__(self):
        return self.name


def loop():
    ij = Player("Indiana Jones")
    mm = Enemy("Mighty Monster")
    print(ij, "vs", mm)
    print("Start!")
    max_tries = 1000
    iter_count = 0
    while True:
        iter_count += 1
        print("iteration=", iter_count, sep='')
        ij.try_move(random.choice(list(Player.MOVES.values())))
        print(ij.state)
        print(ij.cur_state)
        if iter_count >= max_tries or mm.dead:
            print("Game Over")
            return


def main():
    print("Params", "DELTA", DELTA, "GAMMA", GAMMA, "STEP_COST", STEP_COST)
    loop()


if __name__ == '__main__':
    main()
