import numpy as np
import gym
from gym import spaces
import pygame
from constants import *
from init_pygame import init_pygame, load_fonts_and_images


class MinesweeperEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, num_rows=9, num_cols=9, num_mines=10, render_mode="human"):
        """
        Create a minesweeper game.

        Parameters
        ----
        num_rows:   int     num of board's rows
        num_cols:   int     num of board's cols
        num_mines:  int     num of mines on the board

        Returns
        ----
        None
        """

        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_mines = num_mines
        self.board = np.zeros((self.num_rows, self.num_cols), dtype=np.int32)
        self.place_mines()
        self.player_board = np.ones((self.num_rows, self.num_cols), dtype=np.int32) * BrickState.UNKNOWN
        self.observation_space = spaces.Box(low=BrickState.MINE.value, high=BrickState.FLAG.value,
                                            shape=(self.num_rows, self.num_cols), dtype=np.int32)
        self.action_space = spaces.Discrete(self.num_rows * self.num_cols)
        self.reward = None
        self.done = False
        self.info = dict()
        self.first_move = True  # 用于跟踪第一次点击
        self.render_mode = render_mode
        self.stats = {
            'n_win': 0,
            'n_lose': 0,
            'n_progress': 0,
            'n_no_progress': 0,
            'n_guess': 0
        }
        self.button = 'left'
        self.screen = None
        self.smallFont = None
        self.mediumFont = None
        self.largeFont = None
        self.flag_image = None
        self.mine_image = None
        self.reset_button = None
        self.reset_text = None
        self.reset_text_rect = None
        if render_mode == "human":
            self.screen = init_pygame()
            # 重置按钮
            self.smallFont, self.mediumFont, self.largeFont, self.flag_image, self.mine_image = load_fonts_and_images()
            self.reset_button = pygame.Rect(win_width - 5 * BOARD_PADDING - 50, BOARD_PADDING, 100, 40)
            self.reset_text = self.mediumFont.render("reset", True, BLACK)
            self.reset_text_rect = self.reset_text.get_rect(center=self.reset_button.center)

    def reset(self):
        """
        Reset a new game episode.

        Parameters
        ----
        See gym.Env.reset()

        Returns
        ----
        state:  np.array    the initial state of the player's board after reset.
        info:   dict        additional game information.
        """
        self.board = np.zeros((self.num_rows, self.num_cols), dtype=np.int32)
        self.place_mines()
        self.player_board = np.ones((self.num_rows, self.num_cols), dtype=np.int32) * BrickState.UNKNOWN
        self.reward = None
        self.done = False
        self.info = dict()
        self.first_move = True  # 重置游戏后将first_move设为True
        for key in self.stats:
            self.stats[key] = 0
        self.button = 'left'
        return self.player_board

    def step(self, action):
        """
        Take an action in the game environment.

        Parameters
        ----
        action:     np.array    the location on the board where the player wants to take an action.
        Returns
        ----
        next_state: np.array    the current state of the player's board after taking the action.
        reward:     float       the reward received after taking the action.
        done:       bool        a flag indicating whether the game has ended.
        info:       dict        additional game information.
        """
        cell_index = action
        x, y = divmod(cell_index, self.num_cols)
        if self.first_move and self.button == 'left':  # 如果是第一次点击
            if self.board[x, y] == BrickState.MINE:     # 如果第一次点击是地雷
                while self.board[x, y] == BrickState.MINE:  # 重新生成雷区直到不是地雷
                    self.board = np.zeros((self.num_rows, self.num_cols), dtype=np.int32)
                    self.place_mines()
                # print("new board")
                # print(self.board)
            self.first_move = False  # 第一次点击完成后设为False
        # 左键的情况
        if self.button == 'left' and not self.done:
            if not self.is_brick_unknown(x, y) or self.is_brick_flagged(x, y):
                self.reward = -0.5
                self.done = False
                self.info["button"] = "left"
                self.info["status"] = "no_progress"
                self.update_stats("n_no_progress")
            else:
                guess = True if (self.count_neighbour_unknowns(x, y) == 8) else False
                self.player_board[x, y] = self.board[x, y]
                if self.player_board[x, y] == BrickState.EMPTY:
                    self.open_neighbour_bricks(x, y)

                if self.check_game_status() == GameStatus.WIN:
                    self.reward = self.num_rows * self.num_cols
                    self.done = True
                    self.info["button"] = "left"
                    self.info["status"] = "win"
                    self.update_stats("n_win")
                elif self.check_game_status() == GameStatus.LOSE:
                    self.reward = -self.num_rows * self.num_cols
                    self.done = True
                    self.info["button"] = "left"
                    self.info["status"] = "lose"
                    self.update_stats("n_lose")
                elif not guess:
                    self.reward = 1.0
                    self.done = False
                    self.info["button"] = "left"
                    self.info["status"] = "progress"
                    self.update_stats("n_progress")
                else:
                    self.reward = -0.5
                    self.done = False
                    self.info["button"] = "left"
                    self.info["status"] = "guess"
                    self.update_stats("n_guess")
        # 右键的情况(AI扫雷不使用右键)
        elif self.button == 'right' and not self.done:
            if not self.is_brick_unknown(x, y) and not self.is_brick_flagged(x, y):
                self.reward = 0.0
                self.done = False
                self.info["button"] = "right"
                self.info["status"] = "playing"
            elif self.is_brick_flagged(x, y):
                self.player_board[x, y] = BrickState.UNKNOWN
                self.reward = 0.0
                self.done = False
                self.info["button"] = "right"
                self.info["status"] = "playing"
            else:
                self.player_board[x, y] = BrickState.FLAG
                self.reward = 0.0
                self.done = False
                self.info["button"] = "right"
                self.info["status"] = "playing"
        else:
            self.reward = None
            self.done = True
            self.info = {}
        return self.player_board, self.reward, self.done, self.info
        # return self.player_board, self.reward, self.done, self.truncated, self.info

    def render(self, mode='human'):
        """
        Render the current state of the game.

        Depending on the selected render mode, this method either draws the game
        on the screen (for human players) or prepares the game state for
        rendering as an RGB array (for potential machine learning applications).

        Supported render modes:
        - 'human': Displays the game board in a window using Pygame.
        - 'rgb_array': Prepares the game state as an RGB array (currently not implemented).

        Raises
        ------
        ValueError
            If an unsupported render mode is specified.
        """
        if self.render_mode == 'human':
            self.draw_board()
            pygame.display.flip()
        elif self.render_mode == 'rgb_array':
            pass
        else:
            raise ValueError("Invalid render mode. Supported modes are 'human' and 'rgb_array'.")

    def is_brick_unknown(self, x, y):
        """ return true if this is not an already clicked place"""
        return self.player_board[x, y] == BrickState.UNKNOWN

    def is_brick_flagged(self, x, y):
        """ return true if this is a flagged place"""
        return self.player_board[x, y] == BrickState.FLAG

    def is_valid_coordinate(self, x, y):
        """ returns if the coordinate is valid"""
        return 0 <= x < self.num_rows and 0 <= y < self.num_cols

    def check_game_status(self):
        if np.count_nonzero(self.player_board == BrickState.MINE) > 0:
            return GameStatus.LOSE
        elif np.count_nonzero(self.player_board == BrickState.UNKNOWN) + \
                np.count_nonzero(self.player_board == BrickState.FLAG) == self.num_mines:
            return GameStatus.WIN
        else:
            return GameStatus.PLAYING

    def count_neighbour_mines(self, x, y):
        """ return number of mines in neighbour cells given an x-y coordinate """
        neighbour_mines = 0
        for a in range(x - 1, x + 2):
            for b in range(y - 1, y + 2):
                if (a, b) == (x, y):
                    continue
                if self.is_valid_coordinate(a, b):
                    if self.board[a, b] == BrickState.MINE:
                        neighbour_mines += 1
        return neighbour_mines

    def open_neighbour_bricks(self, x, y):
        """ return number of mines in neighbour cells given an x-y coordinate """
        for a in range(x - 1, x + 2):
            for b in range(y - 1, y + 2):
                if (a, b) == (x, y):
                    continue
                if self.is_valid_coordinate(a, b):
                    if self.is_brick_unknown(a, b):
                        self.player_board[a, b] = self.board[a, b]
                        if self.player_board[a, b] == BrickState.EMPTY:
                            self.open_neighbour_bricks(a, b)

    def count_neighbour_unknowns(self, x, y):
        """ return number of UNKNOWN cells in neighbour cells given an x-y coordinate """
        neighbour_unknowns = 0
        for a in range(x - 1, x + 2):
            for b in range(y - 1, y + 2):
                if (a, b) == (x, y):
                    continue
                if self.is_valid_coordinate(a, b):
                    if self.player_board[a, b] == BrickState.UNKNOWN:
                        neighbour_unknowns += 1
                else:
                    neighbour_unknowns += 1
        return neighbour_unknowns

    def place_mines(self):
        """ generate a board, place mines randomly """
        mines_placed = 0
        while mines_placed < self.num_mines:
            i = np.random.randint(0, self.num_rows)
            j = np.random.randint(0, self.num_cols)
            if self.is_valid_coordinate(i, j):
                if self.board[i, j] != BrickState.MINE:
                    self.board[i, j] = BrickState.MINE
                    mines_placed += 1
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if self.board[i, j] != BrickState.MINE:
                    self.board[i, j] = self.count_neighbour_mines(i, j)

    def draw_board(self):
        """ draw the game board using Pygame """
        self.screen.fill(BLACK)
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                rect = pygame.Rect(
                    board_origin[0] + j * brick_size,
                    board_origin[1] + i * brick_size,
                    brick_size, brick_size
                )
                pygame.draw.rect(self.screen, GRAY, rect)
                pygame.draw.rect(self.screen, WHITE, rect, 3)

                if self.player_board[i, j] == BrickState.MINE:
                    self.screen.blit(self.mine_image, rect)
                elif self.player_board[i, j] == BrickState.FLAG:
                    self.screen.blit(self.flag_image, rect)
                elif BrickState.EMPTY <= self.player_board[i, j] <= BrickState.EIGHT:
                    revealed_text = self.smallFont.render(str(self.player_board[i, j]), True, BLACK)
                    revealed_text_rect = revealed_text.get_rect(center=rect.center)
                    self.screen.blit(revealed_text, revealed_text_rect)
        pygame.draw.rect(self.screen, WHITE, self.reset_button)
        self.screen.blit(self.reset_text, self.reset_text_rect)
        self.update_status_panel()

    def update_status_panel(self):
        # 更新显示状态信息
        text = "Win" if (self.done and self.info["status"] == "win") \
            else ("Lose" if self.done and self.info["status"] == "lose" else "Playing")
        COLOR = GREEN if (self.done and self.info["status"] == "win") \
            else (RED if self.done and self.info["status"] == "lose" else GRAY)
        text = "status: " + text
        status_text = self.smallFont.render(text, True, COLOR)
        status_text_rect = status_text.get_rect(center=(win_width - 5 * BOARD_PADDING,
                                                        self.reset_button.height + 2 * BOARD_PADDING))
        self.screen.blit(status_text, status_text_rect)

    def update_stats(self, stat_key: str):
        if stat_key in self.stats:
            self.stats[stat_key] += 1
        else:
            raise KeyError(f"Invalid stat_key: {stat_key}. Must be one of {list(self.stats.keys())}.")
