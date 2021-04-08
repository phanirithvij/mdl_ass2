import os
from collections import OrderedDict
from typing import Iterable
import random
from value_iter import C, R, Ri
import numpy as np

# inital setup
os.makedirs("outputs", exist_ok=True)

# Consts
TEAM_NO = 93
_Y = [1/2, 1, 2][TEAM_NO % 3]
STEP_COST = -10/_Y
GAMMA = 0.999
DELTA = 1e-3

# state mat arrow enemy health actions
# num_states = (5, 3, 4, 2, 5)
num_states = 5*3*4*2*5  # 600
states = np.zeros((5, 3, 4, 2, 5), dtype=np.int)
p_mat = np.zeros((num_states, 10, num_states), dtype=np.float)

# mat -> arr
# arr -> mat

# 0, 0, 0, 0, 0 -> 0
# 599 -> 4, 2, 3, 1, 4


def idx(a, b, c, d, e):
	return 2


# p_mat[w, ..., 2] = 1

print(p_mat[0][0].reshape(5, 3, 4, 2, 5).shape)

# p_mat[]

# state1 (A) -> state2

# exit(0)

step_costs = np.zeros((10,), dtype=np.float)
step_costs[:] = STEP_COST

# 5, 10, 5
move_p_mat = np.array([
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
print(move_p_mat.shape)

mm_att_mat = np.array([
	# R    D
	[0.5, 0.5],  # R na,  a
	[0.2, 0.8],  # D na, na
])


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
		# print(self.arr, self.mapper, key)
		return self.arr[self.mapper[key]]

	def __setitem__(self, key, val):
		self.arr[self.mapper[key]] = val


class PlayerMove:
	LEFT = np.array((-1, 0))
	UP = np.array((0, 1))
	STAY = np.array((0, 0))
	DOWN = np.array((0, -1))
	RIGHT = np.array((1, 0))


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


class MoveIndex:
	LEFT, UP, STAY, DOWN, RIGHT = 0, 1, 2, 3, 4

	_inv_map = DotDict({
		tuple(PlayerMove.LEFT): LEFT,
		tuple(PlayerMove.UP): UP,
		tuple(PlayerMove.STAY): STAY,
		tuple(PlayerMove.DOWN): DOWN,
		tuple(PlayerMove.RIGHT): RIGHT,
	})


class ActionIndex:
	SHOOT, HIT, CRAFT, GATHER, NONE = 5, 6, 7, 8, 9

	_inv_map = DotDict({
		pl_attacks.SHOOT: SHOOT,
		pl_attacks.HIT: HIT,
		pl_actions.CRAFT: CRAFT,
		pl_actions.GATHER: GATHER,
		pl_actions.NONE: NONE,
	})


class ChoiceIndex(MoveIndex, ActionIndex):

	@staticmethod
	def from_choice(choice):
		if type(choice) == int:
			return ActionIndex._inv_map[choice]
		return MoveIndex._inv_map[tuple(choice)]


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
		# all possible choices player could make
		self.choices = []
		self.choices.extend(list(Player.ATTACKS.values()))
		self.choices.extend(list(Player.MOVES.values()))
		self.choices.extend(list(Player.ACTIONS.values()))

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
		prev_state = self.state
		if tuple(self.state) in [Player.STATES.C, Player.STATES.N, Player.STATES.S]:
			if not prob(.85):
				# move/teleport to (E) 15% of time
				self.jump_to_east()
				return prev_state
		# any other move for any state is determined i.e. prob = 1
		self.move(direction)
		return prev_state

	def val_iter(self):
		# for all i
		# if vi - vi-1 < DELTA:
		#     return

		# 600 * 10 * 600
		# None

		# loop over actions
		reward = step_costs[self._action]
		# todo if
		# if
		# reward += 50
		# reward -= 40
		# (vi) = max ((p) (reward +  vi-1 * gamma))

		# old_state (600) 10 (600) new_state

		st_idx = StateIndex.from_state(self.state)
		rs = np.zeros((5,), dtype=np.float)
		fxs = np.zeros((5,), dtype=np.float)
		for i in range(5):
			rs[i] = np.sum(move_p_mat[st_idx][i] * STEP_COST)
			fxs[i] = GAMMA * np.sum(self._values.copy()
									* move_p_mat[st_idx][i])
		lst = rs + fxs
		maxcv = np.max(lst)
		self._values[st_idx] = maxcv
		# self._values[:] = np.round(self._values, 3)

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
			return enemy.bleed(action)

	def craft(self):
		if not np.array_equal(self.state, Player.STATES.N) or self.materials == 0:
			return
		self.materials -= 1
		new_arrows = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
		self.arrows += new_arrows
		return 1, new_arrows

	def gather(self):
		if not np.array_equal(self.state, Player.STATES.S):
			return
		if self.materials == 2:
			return
		if prob(0.75):
			self.materials += 1
			return 1

	# def think(self, enemy):
	#     # thoughts = self.simulate_action(enemy)
	#     # self.val_iter()
	#     # print(
	#     #     f"({pos},{self.materials},{self.arrows},{enemy.state},{enemy.health}):{self.action}=[{self.values[pos]}]")
	#     # self.undo_simulated_action(enemy, *thoughts)
	#     # TODO best action from value iteration
	#     best_action = Player.MOVES.LEFT
	#     return best_action

	def make_move(self, enemy):
		best_action = self.think(enemy)
		if self.stunned:
			# can't make any move for one round
			self.stunned = False
			return
		# perform a random action
		# TODO get valid action
		self._action = random.choice(self.choices)
		self._action = best_action
		self.perform_action(enemy)

	def simulate_action(self, enemy):
		action = self._action
		damage, prev_state, craft_gains, mat_gains = None, None, None, None
		if action in Player.ATTACKS.values():
			damage = self.attack(enemy, action)

		elif action in Player.MOVES.values():
			prev_state = self.try_move(
				random.choice(list(Player.MOVES.values())))
		elif action in Player.ACTIONS.values():
			if action == Player.ACTIONS.CRAFT:
				craft_gains = self.craft()
			elif action == Player.ACTIONS.GATHER:
				mat_gains = self.gather()
			elif action == Player.ACTIONS.NONE:
				pass
		return [damage, prev_state, craft_gains, mat_gains]

	def undo_simulated_action(self, enemy, damage, prev_state, craft_gains, mat_gains):
		action = self._action
		if action in Player.ATTACKS.values():
			if damage is not None:
				enemy.heal(damage)

		elif action in Player.MOVES.values():
			self.state = prev_state
		elif action in Player.ACTIONS.values():
			if action == Player.ACTIONS.CRAFT:
				if craft_gains is not None:
					mat_gone, new_arrows = craft_gains
					self.materials += mat_gone
					self.arrows -= new_arrows
			elif action == Player.ACTIONS.GATHER:
				if mat_gains is not None:
					self.materials -= mat_gains
			elif action == Player.ACTIONS.NONE:
				pass

	def perform_action(self, enemy, undo=False):
		action = self._action
		if action in Player.ATTACKS.values():
			damage = self.attack(enemy, action)
			if undo and damage is not None:
				enemy.heal(damage)

		elif action in Player.MOVES.values():
			prev_state = self.try_move(
				random.choice(list(Player.MOVES.values())))
			if undo:
				self.state = prev_state
		elif action in Player.ACTIONS.values():
			if action == Player.ACTIONS.CRAFT:
				details = self.craft()
				if undo and details is not None:
					mat_gone, new_arrows = details
					self.materials += mat_gone
					self.arrows -= new_arrows
			elif action == Player.ACTIONS.GATHER:
				mat_gains = self.gather()
				if undo and mat_gains is not None:
					self.material -= mat_gains
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
		return damage

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
		for pos, _ in Player.STATES.items():
			for materials in range(3):
				for arrows in range(4):
					for enemy_health in range(0, 125, 25):
						for enemy_state in range(2):
							for action in ij.choices:
								ij._action = action
								for pos_next, _ in Player.STATES.items():
									for materials_next in range(3):
										for arrows_next in range(4):
											for enemy_health_next in range(0, 125, 25):
												for enemy_state_next in range(2):
													idxx = ChoiceIndex.from_choice(
														action)
													reward = step_costs[idxx]
								print(reward)

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
		# print(
		#     f"({ij.cur_state},{ij.materials},{ij.arrows},{mm.state},{mm.health}):{ij.action}=[{ij.values[ij.cur_state]}]")
		if iter_count >= max_tries or mm.dead:
			print("Game Over")
			return


def main():
	print("Params", "DELTA", DELTA, "GAMMA", GAMMA, "STEP_COST", STEP_COST)
	# print(state_mat.shape)
	# print(state_mat[StateIndex.W].shape)
	# print(state_mat[0][0].shape)
	# print(state_mat[0][0][0].shape)
	# print(state_mat[0][0][0][0].shape)
	# print(state_mat[0][0][0][0][0].shape)
	# print(state_mat[0][0][0][0][0][ChoiceIndex.LEFT])
	# state_mat[StateIndex.W, ..., ChoiceIndex.STAY] = 1
	# Task 1

	# Task 2
	# case 1
	# case 2
	step_costs[ChoiceIndex.STAY] = 0
	loop()
	# case 3


if __name__ == '__main__':
	main()
