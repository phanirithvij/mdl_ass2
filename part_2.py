import os

# inital setup
os.makedirs("outputs", exist_ok=True)

# Consts
TEAM_NO = 93
_Y = [1/2, 1, 2][TEAM_NO % 3]
STEP_COST = -10/_Y
GAMMA = 0.999
DELTA = 1e-3

LEFT, RIGHT, UP, DOWN, STAY = 0, 1, 2, 3, 4
ACTIONS = [LEFT, RIGHT, UP, DOWN, STAY]


def main():
    print(DELTA, GAMMA, STEP_COST)


if __name__ == '__main__':
    main()
