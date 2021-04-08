import os
from collections import OrderedDict
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


p_mat = np.array([
    #     W    N     C    S    E
    [
        [.00, .00, .00, .00, .00],  # W, left
        [.00, .00, .00, .00, .00],  # W, up
        [1.0, .00, .00, .00, .00],  # W, stay   -
        [.00, .00, .00, .00, .00],  # W, down
        [.00, .00, 1.0, .00, .00],  # W, right  -
    ],
    [
        [.00, .00, .00, .00, .00],  # N, left
        [.00, .00, .00, .00, .00],  # N, up
        [.00, .85, .00, .00, .15],  # N, stay   -
        [.00, .00, .85, .00, .15],  # N, down   -
        [.00, .00, .00, .00, .00],  # N, right
    ],
    [
        [.85, .00, .00, .00, .15],  # C, left   -
        [.00, .85, .00, .00, .15],  # C, up     -
        [.00, .00, .85, .00, .15],  # C, stay   -
        [.00, .00, .00, .85, .15],  # C, down   -
        [.00, .00, .00, .00, 1.0],  # C, right  -
    ],
    [
        [.00, .00, .00, .00, .00],  # S, left
        [.00, .00, .85, .00, .15],  # S, up     -
        [.00, .00, .00, .85, .15],  # S, stay   -
        [.00, .00, .00, .00, .00],  # S, down
        [.00, .00, .00, .00, .00],  # S, right
    ],
    [
        [.00, .00, 1.0, .00, .00],  # E, left   -
        [.00, .00, .00, .00, .00],  # E, up
        [.00, .00, .00, .00, 1.0],  # E, stay   -
        [.00, .00, .00, .00, .00],  # E, down
        [.00, .00, .00, .00, .00],  # E, right
    ],
])


def prob(p):
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

    def __dir__(self):
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
    UP = np.array((0, 1))
    STAY = np.array((0, 0))
    DOWN = np.array((0, -1))
    RIGHT = np.array((1, 0))


class MoveIndex:
    LEFT, UP, STAY, DOWN, RIGHT = 0, 1, 2, 3, 4


class PlayerState:
    W = np.array((-1, 0))
    N = np.array((0, 1))
    C = np.array((0, 0))
    S = np.array((0, -1))
    E = np.array((1, 0))


class StateIndex:
    W, N, C, S, E = 0, 1, 2, 3, 4
    _inv_map = DotDict({
        tuple(PlayerState.W): W,
        tuple(PlayerState.N): N,
        tuple(PlayerState.C): C,
        tuple(PlayerState.S): S,
        tuple(PlayerState.E): E,
    })

    @staticmethod
    def from_state(state):
        return StateIndex._inv_map[tuple(state)]


# unit vectors for directions
pl_moves = DotDict({
    "LEFT": tuple(PlayerMove.LEFT),
    "UP": tuple(PlayerMove.UP),
    "STAY": tuple(PlayerMove.STAY),
    "DOWN": tuple(PlayerMove.DOWN),
    "RIGHT": tuple(PlayerMove.RIGHT),
})
pl_states = DotDict({
    "W": tuple(PlayerState.W),
    "N": tuple(PlayerState.N),
    "C": tuple(PlayerState.C),
    "S": tuple(PlayerState.S),
    "E": tuple(PlayerState.E),
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

    def __init__(self, name, inital_state=PlayerState.N):
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
    def cur_state(self):
        return _INV_STATE_MAP[tuple(self.state)]

    @property
    def action(self):
        return _INV_ACTIONS_MAP[self._action]

    @property
    def values(self):
        return npdict(self._values, _INV_STATE_MAP_INDEX)

    def jump_to_east(self):
        self.state = PlayerState.E

    def try_move(self, direction):
        if tuple(self.state) in [Player.STATES.C, Player.STATES.N, Player.STATES.S]:
            if not prob(.85):
                # move/teleport to (E) 15% of time
                self.jump_to_east()
                return
        # any other move for any state is determined i.e. prob = 1
        self.move(direction)

    def val_iter(self):
        st_idx = StateIndex.from_state(self.state)
        rs = np.zeros((5,), dtype=np.float)
        fxs = np.zeros((5,), dtype=np.float)
        for i in range(5):
            rs[i] = np.sum(p_mat[st_idx][i] * STEP_COST)
            fxs[i] = GAMMA * np.sum(self._values.copy() * p_mat[st_idx][i])
        lst = rs + fxs
        maxcv = np.max(lst)
        self._values[st_idx] = maxcv
        # self._values[:] = np.round(self._values, 9)

    def get_wrecked(self):
        if tuple(self.state) not in [Player.STATES.C, Player.STATES.E]:
            return
        self.arrows = 0
        self.reward -= 40
        self.stunned = True

    def attack(self, enemy, action):
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
        self.val_iter()
        if self.stunned:
            # can't make any move for one round
            self.stunned = False
            return
        if not hasattr(self, 'choices'):
            # all possible choices player could make
            self.choices = []
            self.choices.extend(list(Player.ATTACKS.values()))
            self.choices.extend(list(Player.MOVES.values()))
            self.choices.extend(list(Player.ACTIONS.values()))
        # perform a random action
        self._action = random.choice(self.choices)
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

    def __init__(self, name):
        self.name = name
        self.health = 100
        self.state = Enemy.STATES.D

    def bleed(self, damage = 25):
        self.health -= damage

    def heal(self, health = 25):
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

    def try_attack(self, player):
        if prob(0.5):
            # attack
            player.get_wrecked()
            self.heal()
            self.rest()

    def make_move(self, player):
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

