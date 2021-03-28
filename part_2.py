import os
from collections import OrderedDict
from typing import Iterable
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


class DotDict(OrderedDict):
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


class npdict():
    """To be able to access np array with strings"""

    def __init__(self, nparr, one_one_mapper):
        self.arr = nparr
        self.mapper = one_one_mapper

    def __getitem__(self, key):
        return self.arr[self.mapper[key]]

    def __setitem__(self, key, val):
        self.arr[self.mapper[key]] = val


class PlayerMove:
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
    "LEFT": tuple(PlayerMove.LEFT),
    "RIGHT": tuple(PlayerMove.RIGHT),
    "UP": tuple(PlayerMove.UP),
    "DOWN": tuple(PlayerMove.DOWN),
    "STAY": tuple(PlayerMove.STAY)
})
pl_states = DotDict({
    "W": tuple(PlayerState.W),
    "E": tuple(PlayerState.E),
    "N": tuple(PlayerState.N),
    "S": tuple(PlayerState.S),
    "C": tuple(PlayerState.C)
})

pl_attacks = DotDict({"SHOOT": 25, "HIT": 50})

_INV_STATE_MAP = {pl_states[k]: k for k in pl_states}
_INV_STATE_MAP_INDEX = {k: i for i, k in enumerate(pl_states)}

pl_actions = DotDict({"CRAFT": 2, "GATHER": 3, "NONE": 4})

_INV_ACTIONS_MAP = {pl_actions[k]: k for k in pl_actions}
_INV_ACTIONS_MAP.update({pl_moves[k]: k for k in pl_moves})
_INV_ACTIONS_MAP.update({pl_attacks[k]: k for k in pl_attacks})


class Player(object):
    MOVES = pl_moves
    STATES = pl_states
    ATTACKS = pl_attacks
    ACTIONS = pl_actions

    def __init__(self, name: str, inital_state=PlayerState.N):
        self.name = name
        self.state = inital_state
        self._action = Player.ACTIONS.NONE
        self.arrows = 0
        self.materials = 0
        self.reward = 0
        self.stunned = False
        self._values = np.zeros((5,), dtype=np.float)

    def move(self, direction):
        # add direction to state
        self.state += direction
        if np.sum(np.abs(self.state)) > 1:
            # out of bounds, illegal move -> undo move
            self.state -= direction

    @property
    def cur_state(self) -> str:
        return _INV_STATE_MAP[tuple(self.state)]

    @property
    def action(self) -> str:
        return _INV_ACTIONS_MAP[self._action]

    @property
    def values(self):
        return npdict(self._values, _INV_STATE_MAP_INDEX)

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

    def get_wrecked(self):
        if tuple(self.state) not in [Player.STATES.C, Player.STATES.E]:
            return
        self.arrows = 0
        self.reward -= 40
        self.stunned = True

    def attack(self, enemy, action: int):
        success_prob = 0
        if action == Player.ATTACKS.SHOOT:
            # check arrows
            if self.arrows == 0:
                return
            self.arrows -= 1
            if np.array_equal(self.state, PlayerState.C):
                success_prob = 0.5
            elif np.array_equal(self.state, PlayerState.W):
                success_prob = 0.25
            elif np.array_equal(self.state, PlayerState.E):
                success_prob = 0.9
        elif action == Player.ATTACKS.HIT:
            if np.array_equal(self.state, PlayerState.C):
                success_prob = 0.1
            elif np.array_equal(self.state, PlayerState.E):
                success_prob = 0.2

        # deal appropriate damage with success_probability
        if prob(success_prob):
            enemy.bleed(action)

    def craft(self):
        if not np.array_equal(self.state, Player.STATES.N) or self.materials == 0:
            return
        self.materials -= 1
        self.arrows += np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])

    def gather(self):
        if not np.array_equal(self.state, Player.STATES.S):
            return
        if prob(0.75):
            if self.materials == 0:
                self.materials += 1

    def make_move(self, enemy):
        if self.stunned:
            # can't make any move for one round
            self.stunned = False
            return
        choices = []
        choices.extend(list(Player.ATTACKS.values()))
        choices.extend(list(Player.MOVES.values()))
        choices.extend(list(Player.ACTIONS.values()))
        self._action = random.choice(choices)
        self.perform_action(enemy)

    def perform_action(self, enemy):
        action = self._action
        if action in Player.ATTACKS.values():
            self.attack(enemy, action)
        elif action in Player.MOVES.values():
            self.try_move(random.choice(list(Player.MOVES.values())))
        elif action in Player.ACTIONS.values():
            if action == Player.ACTIONS.CRAFT:
                self.craft()
            elif action == Player.ACTIONS.GATHER:
                self.gather()
            elif action == Player.ACTIONS.NONE:
                pass

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

    def rest(self):
        self.state = Enemy.STATES.D

    def think(self):
        if prob(0.2):
            self.get_ready()

    def try_attack(self, player: Player):
        if prob(0.5):
            # attack
            player.get_wrecked()
            self.heal()
            self.rest()

    def make_move(self, player: Player):
        self.think()
        if self.state == Enemy.STATES.R:
            self.try_attack(player)

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
        mm.make_move(ij)
        ij.make_move(mm)
        # print(ij.cur_state)
        # print(ij.materials)
        # print(ij.arrows)
        # print(mm.state)
        # print(mm.health)
        # print(ij.action)
        # print(ij.values[ij.cur_state])
        print(
            f"({ij.cur_state},{ij.materials},{ij.arrows},{mm.state},{mm.health}):{ij.action}=[{ij.values[ij.cur_state]}]")
        if iter_count >= max_tries or mm.dead:
            print("Game Over")
            return


def main():
    print("Params", "DELTA", DELTA, "GAMMA", GAMMA, "STEP_COST", STEP_COST)
    loop()


if __name__ == '__main__':
    main()
