import sys
import time
import torch
import numpy as np

from lib import models
from lib.wrappers import CustomObservationWrapper
from minesweeper_pygame import *

game_mode = "computer"
if game_mode == "computer":
    render_mode = "human"
    env = MinesweeperEnv(HEIGHT, WIDTH, NUM_MINES, render_mode)
    env = CustomObservationWrapper(env, encoding_type="one_hot")
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = r"./models/CNN-1725462830/Minesweeper-v0-best_win_rate.dat"  # 替换为你实际的模型路径
    net = models.Model(env.observation_space.shape, env.action_space.n).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()  # 设置模型为评估模式

    episodes = 100
    nWin = 0
    delay = True  # 这里可以设置为 False 以禁用显示
    for episode in range(1, episodes + 1):
        if render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
        state = env.reset()
        done = False
        episode_rewards = 0  # 记录每个回合的得分
        episode_steps = 0  # 记录每个回合的时间步数
        info = None  # 初始化 info 变量
        first_move = True  # 初始化 first_move 标志
        while not done:
            # 随机选择任意的单元格(完全随机)
            # action = env.action_space.sample()  # this is where you would insert your policy

            # 随机选择未知的单元格, 适用于one_hot和condensed编码
            # board = state[-1].reshape(1, env.action_space.n)
            # unknown = [i for i, x in enumerate(board[0]) if x == 1]
            # action = np.random.choice(unknown)

            if first_move:
                action = 0  # 第一次移动时点击左上角
                first_move = False  # 更新 first_move 标志
            # elif np.random.random() < 0.01:
            #     board = state[-1].reshape(1, env.action_space.n)
            #     unknown = [i for i, x in enumerate(board[0]) if x == 1]
            #     action = np.random.choice(unknown)
            else:
                # 使用模型预测动作
                with torch.no_grad():
                    input_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    output_tensor = net(input_tensor)
                    probability_array = output_tensor.cpu().detach().numpy()
                    probability_array = np.squeeze(probability_array)

                    mask = state[-1].reshape(-1) == 0
                    probability_array_flat = probability_array.reshape(-1)
                    probability_array_flat[mask] = np.inf
                    min_index = np.argmin(probability_array_flat)
                    action = None if probability_array_flat[min_index] == np.inf else min_index

            env.button = 'left'
            next_state, reward, done, info = env.step(action)
            state = next_state
            episode_rewards += reward
            episode_steps += 1
            env.render()
            if delay:
                time.sleep(0.5)
        if delay:
            time.sleep(2)
        if info["status"] == "win":
            nWin += 1
        print(f"Episode:{episode} Rewards:{episode_rewards:.2f} Steps:{episode_steps} Total Wins:{nWin} ")
        print(f"episode stats: ", env.stats)
    env.close()
