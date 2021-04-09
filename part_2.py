import os
from collections import OrderedDict
from typing import Iterable
import random
import numpy as np
import sys

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
# num_states = 5*3*4*2*5  # 600
# p_mat = np.zeros((num_states, 10, num_states), dtype=np.float)

# mat -> arr
# arr -> mat

# 0, 0, 0, 0, 0 -> 0
# 599 -> 4, 2, 3, 1, 4


# def idx(a, b, c, d, e):
# 	return 2


# p_mat[w, ..., 2] = 1

# print(p_mat[0][0].reshape(5, 3, 4, 2, 5).shape)

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
# print(move_p_mat.shape)

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
_INV_STATE_MAP_INDEX_STRS = {i: k for i, k in enumerate(pl_states)}
_INV_STATE_MAP_NP_INDEX = {i: np.array(
	pl_states[k]) for i, k in enumerate(pl_states)}

pl_actions = DotDict({"CRAFT": 1221, "GATHER": 3233, "NONE": 41212})

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

	_map = DotDict({
		LEFT: tuple(PlayerMove.LEFT),
		UP: tuple(PlayerMove.UP),
		STAY: tuple(PlayerMove.STAY),
		DOWN: tuple(PlayerMove.DOWN),
		RIGHT: tuple(PlayerMove.RIGHT),
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


	_map = DotDict({
		SHOOT:		pl_attacks.SHOOT,
		HIT:		pl_attacks.HIT,
		CRAFT:		pl_actions.CRAFT,
		GATHER:		pl_actions.GATHER,
		NONE:		pl_actions.NONE,
	})


class ChoiceIndex(MoveIndex, ActionIndex):

	_inv_map_strs = DotDict({
		MoveIndex.LEFT: "LEFT",
		MoveIndex.UP: "UP",
		MoveIndex.STAY: "STAY",
		MoveIndex.DOWN: "DOWN",
		MoveIndex.RIGHT: "RIGHT",
		ActionIndex.SHOOT: "SHOOT",
		ActionIndex.HIT: "HIT",
		ActionIndex.CRAFT: "CRAFT",
		ActionIndex.GATHER: "GATHER",
		ActionIndex.NONE: "NONE",
	})

	@staticmethod
	def from_choice(choice):
		if type(choice) == int:
			return ActionIndex._inv_map[choice]
		return MoveIndex._inv_map[tuple(choice)]
	
	@staticmethod
	def to_choice(choice):
		if choice >= 5:
			return ActionIndex._map[choice]
		return MoveIndex._map[choice]


	@staticmethod
	def str(choice):
		strx = ""
		try:
			strx = ChoiceIndex._inv_map_strs[choice]
		except KeyError:
			print(choice)
			strx = "ERROR"
		return strx


class Player(object):
	MOVES = pl_moves
	STATES = pl_states
	ATTACKS = pl_attacks
	ACTIONS = pl_actions

	def __init__(self, name: str, inital_state=PlayerState.N, enemy=None):
		self.enemy = enemy
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

	def move(self, direction, simulate=False):
		# add direction to state
		self.state += direction
		valid = np.sum(np.abs(self.state)) <= 1
		if simulate:
			self.state -= direction
			return valid
		if np.sum(np.abs(self.state)) > 1:
			# out of bounds, illegal move -> undo move
			self.state -= direction


	def valid_move(self, direction, new_state_pos):
		if type(direction) == int or type(new_state_pos) == int:
			print(new_state_pos, direction)
		return tuple(new_state_pos - direction) == tuple(self.state) or (tuple(self.state) != Player.STATES.W and tuple(new_state_pos) == Player.STATES.E)

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

	def try_move(self, direction: tuple, simulate=False):
		prev_state = self.state
		can_move = self.move(direction, simulate)
		if can_move:
			return prev_state

	def try_next_move(self, direction: tuple, new_state_pos):
		# can_move = self.valid_move(direction, new_state_pos)
		# if not can_move:
		# 	return 0
		prev_state_idx = StateIndex.from_state(self.state)
		new_state_idx = StateIndex.from_state(new_state_pos)
		probx = move_p_mat[prev_state_idx][ChoiceIndex.from_choice(direction)][new_state_idx]
		# if new_state_idx == StateIndex.E:
			# if probx == 0.15:
			# 	print("Tele", "E")
			# 	print(prev_state_idx, ChoiceIndex.from_choice(direction), new_state_idx, probx)
		return probx

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

	def get_wrecked(self, simulate=False):
		if tuple(self.state) not in [Player.STATES.C, Player.STATES.E]:
			return
		self.arrows = 0
		self.reward -= 40
		self.stunned = True
		if simulate:
			return True

	def check_attack(self, action: int, new_state):
		if not np.array_equal(new_state.state, self.state):
			return 0

		probx = 0
		if action == Player.ATTACKS.SHOOT:
			if new_state.arrows != self.arrows - 1:
				return 0

			if np.array_equal(self.state, PlayerState.C):
				if self.enemy.health == new_state.enemy.health:
					probx = 0.5
				elif self.enemy.health == new_state.enemy.health + action:
					probx = 0.5
			elif np.array_equal(self.state, PlayerState.W):
				if self.enemy.health == new_state.enemy.health:
					probx = 0.75
				elif self.enemy.health == new_state.enemy.health + action:
					probx = 0.25
			elif np.array_equal(self.state, PlayerState.E):
				if self.enemy.health == new_state.enemy.health:
					probx = 0.1
				elif self.enemy.health == new_state.enemy.health + action:
					probx = 0.9
			else:
				probx = 0

		elif action == Player.ATTACKS.HIT:
			if np.array_equal(self.state, PlayerState.C):
				if self.enemy.health == new_state.enemy.health:
					probx = 0.9
				elif self.enemy.health == new_state.enemy.health + action:
					probx = 0.1
			elif np.array_equal(self.state, PlayerState.E):
				if self.enemy.health == new_state.enemy.health:
					probx = 0.8
				elif self.enemy.health == new_state.enemy.health + action:
					probx = 0.2
			else:
				probx = 0

		# print("success_prob", success_prob)
		return probx
		# deal appropriate damage with success_probability

	def attack(self, action: int, simulate=False):
		success_prob = None
		if action == Player.ATTACKS.SHOOT:
			# check arrows
			if self.arrows == 0:
				return
			if not simulate:
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

		if simulate:
			return success_prob
		# deal appropriate damage with success_probability
		if prob(success_prob):
			return self.enemy.bleed(action)

	def craft(self, simulate=False):
		if not np.array_equal(self.state, Player.STATES.N) or self.materials == 0:
			return
		if simulate:
			return True
		self.materials -= 1
		new_arrows = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
		self.arrows += new_arrows
		return 1, new_arrows

	def check_craft(self, new_state):
		if not np.array_equal(new_state.state, Player.STATES.N):
			return 0
		if self.materials != new_state.materials+1:
			return 0
		arrow_diff = abs(self.arrows - new_state.arrows)
		if arrow_diff == 0:
			return 0
		return [0.5, 0.35, 0.15][arrow_diff-1]

	def gather(self, simulate=False):
		if not np.array_equal(self.state, Player.STATES.S) or self.materials == 2:
			return
		if simulate:
			return 0.75
		if prob(0.75):
			self.materials += 1
			return 1

	def check_gather(self, new_state):
		if not np.array_equal(new_state.state, Player.STATES.S):
			return 0
		if new_state.materials == self.materials + 1:
			return 0.75
		elif new_state.materials == self.materials:
			return 0.25

		return 0

	def think(self):
		# thoughts = self.try_action(self.enemy)
		# self.val_iter()
		# print(
		#     f"({pos},{self.materials},{self.arrows},{self.enemy.state},{self.enemy.health}):{self.action}=[{self.values[pos]}]")
		# self.undo_simulated_action(self.enemy, *thoughts)
		# TODO best action from value iteration
		best_action = Player.MOVES.LEFT
		return best_action

	def make_move(self, simulate=False):
		best_action = self.think()
		if self.stunned:
			# can't make any move for one round
			self.stunned = False
			return
		# perform a random action
		# TODO get valid action
		self._action = random.choice(self.choices)
		self._action = best_action
		self.perform_action()

	def check_new_state(self, new_state):
		action = self._action
		damage_prob, prev_state_prob, craft_prob, mat_gains_prob = 0, 0, 0, 0
		probx = 0
		if action in Player.ATTACKS.values():
			damage_prob = self.check_attack(action, new_state)
			# print("damage_prob", damage_prob)
			probx = damage_prob
		elif action in Player.MOVES.values():
			prev_state_prob = self.try_next_move(action, new_state.state)
			# print("prev_state_prob", prev_state_prob)
			probx = prev_state_prob
		elif action in Player.ACTIONS.values():
			if action == Player.ACTIONS.CRAFT:
				craft_prob = self.check_craft(new_state)
				# print("craft_prob", craft_prob)
				probx = craft_prob
			elif action == Player.ACTIONS.GATHER:
				mat_gains_prob = self.check_gather(new_state)
				# print("mat_gains_prob", mat_gains_prob)
				probx = mat_gains_prob
			elif action == Player.ACTIONS.NONE:
				# print("none", 0)
				probx = 0

		if self.enemy.state == Enemy.STATES.R:
			if new_state.enemy.state == Enemy.STATES.R:
				probx = probx * 0.5
			else:
				# nothing is successful so prob = 0.5
				probx = 0.5
				if new_state.arrows != 0:
					return 0
		elif self.enemy.state == Enemy.STATES.D:
			if new_state.enemy.state == Enemy.STATES.R:
				probx = probx * 0.2
			else:
				probx = probx * 0.8

		return probx

	def try_action(self):
		action = self._action
		damage, prev_state, craft_gains, mat_gains = None, None, None, None
		if action in Player.ATTACKS.values():
			damage = self.attack(action, simulate=True)
			if damage is None:
				return
		elif action in Player.MOVES.values():
			prev_state = self.try_move(action, simulate=True)
			if prev_state is None:
				return
		elif action in Player.ACTIONS.values():
			if action == Player.ACTIONS.CRAFT:
				craft_gains = self.craft(simulate=True)
				if craft_gains is None:
					return
			elif action == Player.ACTIONS.GATHER:
				mat_gains = self.gather(simulate=True)
				if mat_gains is None:
					return
			elif action == Player.ACTIONS.NONE:
				return
		return [damage, prev_state, craft_gains, mat_gains]

	def undo_simulated_action(self, damage, prev_state, craft_gains, mat_gains):
		action = self._action
		if action in Player.ATTACKS.values():
			if damage is not None:
				self.enemy.heal(damage)
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

	def perform_action(self, undo=False):
		action = self._action
		if action in Player.ATTACKS.values():
			damage = self.attack(action)
			if undo and damage is not None:
				self.enemy.heal(damage)

		elif action in Player.MOVES.values():
			prev_state = self.try_move(action)
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
		self.player = None
		self.name = name
		self.health = 100
		self.state = Enemy.STATES.D

	def encounter_player(self, player):
		self.player = player

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

	def try_attack(self, player: Player, simulate=False):
		if simulate:
			self.heal()
			self.rest()
			return player.get_wrecked(simulate)
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

	mm = Enemy("Mighty Monster")
	mm_next = Enemy("Mighty Monster")

	ij = Player("Indiana Jones", enemy=mm)
	ij_next = Player("Indiana Jones", enemy=mm)

	mm.encounter_player(ij)
	mm_next.encounter_player(ij_next)

	# ij.state = np.array(Player.STATES.W)
	# print(ij.move(Player.MOVES.DOWN, simulate=True))
	# return

	# print(ij, "vs", mm)
	# print("Start!")
	max_tries = 1000
	iter_count = 0

	utilities = np.zeros((5, 3, 4, 2, 5), dtype=np.float)
	actions = np.zeros((5, 3, 4, 2, 5), dtype=np.int)

	minus_infs = utilities.copy()
	minus_infs[:] = -np.inf

	stop_err = -np.inf

	# print(ij.choices.__len__())

	while True:
		stop_err = -np.inf

		new_utilities = minus_infs.copy()
		new_utilities[:, :, :, :, 0] = 0

		iter_count += 1
		print("iteration=", iter_count, sep='')
		for pos in range(5):
			for materials in range(3):
				for arrows in range(4):
					for enemy_state in range(2):
						for enemy_health in range(25, 125, 25):
							for action in ij.choices:
								ij._action = action
								ij.arrows = arrows
								ij.materials = materials
								ij.state = _INV_STATE_MAP_NP_INDEX[pos]
								mm.health = enemy_health
								mm.state = enemy_state

								thoughts = ij.try_action()
								if thoughts is None:
									continue

								ij._action = action
								ij.arrows = arrows
								ij.materials = materials
								ij.state = _INV_STATE_MAP_NP_INDEX[pos]
								mm.health = enemy_health
								mm.state = enemy_state

								val = 0
								for pos_next in range(5):
									for materials_next in range(3):
										for arrows_next in range(4):
											for enemy_state_next in range(2):
												for enemy_health_next in range(0, 125, 25):
													idxx = ChoiceIndex.from_choice(
														action)
													reward = step_costs[idxx]

													ij_next._action = action
													ij_next.arrows = arrows_next
													ij_next.materials = materials_next
													ij_next.state = _INV_STATE_MAP_NP_INDEX[pos_next]
													mm_next.health = enemy_health_next
													mm_next.state = enemy_state_next

													# rewards
													if enemy_health_next == 0:
														reward += 50

													# attacked -40
													if ij.enemy.state == Enemy.STATES.R:
														if mm_next.state == Enemy.STATES.D:
															reward -= 40

													next_idexr = pos_next, materials_next, arrows_next, enemy_state_next, (
														enemy_health_next // 25)

													probx = ij.check_new_state(ij_next)
													# if probx == 0:
													# 	continue
													valx = probx * (
														reward + GAMMA * utilities[next_idexr])
													val += valx
													# print(probx, reward, valx, next_idexr, val)

								idexr = pos, materials, arrows, enemy_state, (
									enemy_health // 25)
								# print(new_utilities[idexr], val)
								if val >= new_utilities[idexr]:
									new_utilities[idexr] = val
									# print("-"*10, val)

								curr_err = np.abs(
									utilities[idexr] - new_utilities[idexr])
								if stop_err < curr_err:
									stop_err = curr_err

		utilities = new_utilities.copy()
		actions[:] = -1e30
		actions[:, :, :, :, 0] = ChoiceIndex.NONE
		future_utilities = minus_infs.copy()
		future_utilities[:, :, :, :, 0] = 0

		for pos in range(5):
			for materials in range(3):
				for arrows in range(4):
					for enemy_state in range(2):
						for enemy_health in range(25, 125, 25):
							for action in ij.choices:
								ij._action = action
								ij.arrows = arrows
								ij.materials = materials
								ij.state = _INV_STATE_MAP_NP_INDEX[pos]
								mm.health = enemy_health
								mm.state = enemy_state

								thoughts = ij.try_action()
								if thoughts is None:
									continue

								ij._action = action
								ij.arrows = arrows
								ij.materials = materials
								ij.state = _INV_STATE_MAP_NP_INDEX[pos]
								mm.health = enemy_health
								mm.state = enemy_state

								val = 0
								for pos_next in range(5):
									for materials_next in range(3):
										for arrows_next in range(4):
											for enemy_state_next in range(2):
												for enemy_health_next in range(0, 125, 25):
													idxx = ChoiceIndex.from_choice(
														action)
													reward = step_costs[idxx]

													ij_next._action = action
													ij_next.arrows = arrows_next
													ij_next.materials = materials_next
													ij_next.state = _INV_STATE_MAP_NP_INDEX[pos_next]
													mm_next.health = enemy_health_next
													mm_next.state = enemy_state_next

													# rewards
													if enemy_health_next == 0:
														reward += 50

													if ij.enemy.state == Enemy.STATES.R:
														if mm_next.state == Enemy.STATES.D:
															reward -= 40

													next_idexr = pos_next, materials_next, arrows_next, enemy_state_next, (
														enemy_health_next // 25)

													probx = ij.check_new_state(ij_next)
													# if probx == 0:
													# 	continue
													val += probx * (
														reward + GAMMA * utilities[next_idexr])

								idxer = pos, materials, arrows, enemy_state, enemy_health // 25
								if val >= future_utilities[idxer]:
									future_utilities[idxer] = val
									actions[idxer] = ChoiceIndex.from_choice(
										action)

		for pos in range(5):
			for materials in range(3):
				for arrows in range(4):
					for enemy_state in range(2):
						for enemy_health in range(0, 125, 25):
							idxer = pos, materials, arrows, enemy_state, enemy_health // 25
							action = ChoiceIndex.to_choice(actions[idxer])
							ij._action = action

							ij.arrows = arrows
							ij.materials = materials
							ij.state = _INV_STATE_MAP_NP_INDEX[pos]
							mm.health = enemy_health
							mm.state = enemy_state

							thoughts = ij.try_action()
							if thoughts is None:
								continue
							print(
								f"({_INV_STATE_MAP_INDEX_STRS[pos]},{materials},{arrows},{enemy_state},{enemy_health}):{ChoiceIndex.str(actions[idxer])}=[{utilities[idxer]}]")
								# f"({_INV_STATE_MAP_INDEX_STRS[pos]},{materials},{arrows},{enemy_state},{enemy_health}):{ChoiceIndex.str(actions[idxer])}=[{utilities[idxer]}, {new_utilities[idxer]}, {future_utilities[idxer]}]")

		if (iter_count >= max_tries) or (stop_err <= DELTA):
			return


def main():
	global GAMMA
	print("Params", "DELTA", DELTA, "GAMMA", GAMMA, "STEP_COST", STEP_COST)
	# Task 1

	original = sys.stdout

	# Task 2

	global move_p_mat
	global step_costs

	move_p_mat_backup = move_p_mat.copy()
	step_costs_backup = step_costs.copy()
	gamma_backup = GAMMA
	# case 1

	sys.stdout = open('./outputs/part_2_task_2.1_trace.txt', 'w')
	move_p_mat[StateIndex.E][0] = np.array([1.0, .00, .0, .00, .00])
	loop()
	sys.stdout.close()
	sys.stdout = original
	move_p_mat = move_p_mat_backup.copy()

	# case 2
	# step_costs[ChoiceIndex.STAY] = 0
	sys.stdout = open('./outputs/part_2_task_2.2_trace.txt', 'w')
	step_costs[ChoiceIndex.STAY] = 0
	loop()
	sys.stdout.close()
	sys.stdout = original
	step_costs = step_costs_backup.copy()

	# case 3
	sys.stdout = open('./outputs/part_2_task_2.3_trace.txt', 'w')
	GAMMA = 0.25
	loop()
	sys.stdout.close()
	sys.stdout = original
	GAMMA = gamma_backup


if __name__ == '__main__':
	main()
