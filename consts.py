TEAM_NO = 93
_Y = [1/2, 1, 2][TEAM_NO%3]
STEP_COST = -10/_Y
GAMMA = 0.999
DELTA = 1e-3

if __name__ == '__main__':
    print(TEAM_NO, STEP_COST, GAMMA, DELTA)
