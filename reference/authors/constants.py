# constants.py
from enum import IntEnum


# 定义雷区状态
class BrickState(IntEnum):
    MINE = -2
    UNKNOWN = -1
    EMPTY = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    FLAG = 9


class GameStatus(IntEnum):
    LOSE = -1
    PLAYING = 0
    WIN = 1


class Button(IntEnum):
    LEFT = 1
    RIGHT = 3


# 设定游戏难度
difficulty_index = 3
if difficulty_index == 0:
    HEIGHT, WIDTH, NUM_MINES = 9, 9, 10
elif difficulty_index == 1:
    HEIGHT, WIDTH, NUM_MINES = 16, 16, 40
elif difficulty_index == 2:
    HEIGHT, WIDTH, NUM_MINES = 16, 30, 99
else:
    HEIGHT, WIDTH, NUM_MINES = 6, 6, 4

# 设定界面尺寸
BOARD_PADDING = 20
brick_size = 30
win_width = BOARD_PADDING + WIDTH * brick_size + BOARD_PADDING * 10
win_height = BOARD_PADDING + HEIGHT * brick_size + BOARD_PADDING
board_origin = (BOARD_PADDING, BOARD_PADDING)
board_width = WIDTH * brick_size
board_height = HEIGHT * brick_size
win_size = (win_width, win_height)

# 定义颜色
BLACK = (0, 0, 0)
GRAY = (180, 180, 180)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# net settings
CONV_UNITS = 64  # number of neurons in each conv layer
DENSE_UNITS = 256  # number of neurons in fully connected dense layer
