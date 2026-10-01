# init_pygame.py
import pygame
from constants import *


def init_pygame():
    pygame.init()
    screen = pygame.display.set_mode(win_size)
    return screen


def load_fonts_and_images():
    # 设定字体
    # noinspection SpellCheckingInspection
    OPEN_SANS = "assets/fonts/simkai.ttf"
    smallFont = pygame.font.Font(OPEN_SANS, 16)
    mediumFont = pygame.font.Font(OPEN_SANS, 32)
    largeFont = pygame.font.Font(OPEN_SANS, 48)
    # 加载图像
    flag_image = pygame.image.load("assets/images/flag.png")
    flag_image = pygame.transform.scale(flag_image, (brick_size, brick_size))
    mine_image = pygame.image.load("assets/images/mine.png")
    mine_image = pygame.transform.scale(mine_image, (brick_size, brick_size))

    return smallFont, mediumFont, largeFont, flag_image, mine_image
