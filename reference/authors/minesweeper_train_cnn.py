import os
import argparse
import copy

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
from datetime import datetime

from tensorboardX import SummaryWriter
from lib import models
from minesweeper_pygame import *
from lib.wrappers import CustomObservationWrapper

ENV_NAME = "Minesweeper-v0"
REWARD_BOUND = 30.0
WIN_RATE_BOUND = 0.95
AGG_STATS_EVERY = 100
SAVE_MODEL_EVERY = 500
LEARNING_RATE = 1e-5

model_dir = f"./models/CNN-{int(time.time())}"
os.makedirs(model_dir, exist_ok=True)
log_dir = f"./logs/CNN-{int(time.time())}"
os.makedirs(log_dir, exist_ok=True)


def _convert_board(board):
    label = np.where(board == BrickState.MINE, 1, 0).reshape(-1)
    return label


# 定义训练类
class Agent:
    def __init__(self, env_cnn, net_cnn, device_cnn, model_dir_cnn, num_samples=320, epochs=10):
        self.env = env_cnn
        self.model = net_cnn
        self.device = device_cnn
        self.state = env_cnn.reset()
        self.episode_reward = 0.0
        self.moves_made = 0
        self.cells_left = 0
        self.guesses_made = 0
        self.games_played = 0
        self.games_won = 0
        self.num_samples = num_samples
        self.epochs = epochs
        self.current_sample = 0
        self.features = np.zeros((num_samples, self.state.shape[0], self.env.num_rows, self.env.num_cols))
        self.labels = np.zeros((num_samples, self.env.action_space.n))
        self.optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
        self.stats_file = os.path.join(log_dir, "stats.txt")  # 创建统计文件路径
        self.model_dir = model_dir_cnn
        # 添加标题栏
        if not os.path.exists(self.stats_file):
            with open(self.stats_file, "w") as f:
                f.write("games_played;games_won;moves_made;cells_left;guessed_made;timestamp;\n")

    def _reset(self):
        self.state = env.reset()
        self.episode_reward = 0.0
        self.moves_made = 0
        self.cells_left = 0
        self.guesses_made = 0

    def play_step(self):
        done_episode_reward = None
        done_moves_made = None
        done_cells_left = None
        done_guesses_made = None
        done_stats = None
        input_tensor = torch.tensor(self.state, dtype=torch.float32).unsqueeze(0).to(self.device)
        output_tensor = self.model(input_tensor)
        probability_array = output_tensor.cpu().detach().numpy()
        probability_array = np.squeeze(probability_array)

        mask = self.state[-1].reshape(-1) == 0
        probability_array[mask] = np.inf
        min_index = np.argmin(probability_array)
        action = None if probability_array[min_index] == np.inf else min_index

        self.env.button = 'left'
        next_state, step_reward, is_done, _ = self.env.step(action)
        self.episode_reward += step_reward
        self.state = next_state
        self.moves_made += 1
        self.increment_sample_size(self.state, _convert_board(self.env.board))
        if is_done:
            self.games_played += 1
            self.games_won += self.env.stats['n_win']
            self.guesses_made += self.env.stats['n_guess']
            self.cells_left = np.sum((self.state[-1] == 1) & (self.env.board != BrickState.MINE))
            done_episode_reward = self.episode_reward
            done_guesses_made = self.guesses_made - 1
            done_moves_made = self.moves_made
            done_cells_left = self.cells_left
            done_stats = copy.deepcopy(self.env.stats)
            self._reset()
        return done_episode_reward, done_moves_made, done_cells_left, done_guesses_made, \
            self.games_played, self.games_won, done_stats

    def update_model(self):
        features_tensor = torch.tensor(self.features, dtype=torch.float32).to(self.device)
        labels_tensor = torch.tensor(self.labels, dtype=torch.float32).to(self.device)
        dataset = TensorDataset(features_tensor, labels_tensor)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=True)  # 使用 DataLoader 打乱数据
        for epoch in range(self.epochs):
            for batch_features, batch_labels in data_loader:  # 迭代 DataLoader 中的批次
                batch_features, batch_labels = batch_features.to(self.device), batch_labels.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(batch_features)
                loss = self.model.loss(output, batch_labels)
                loss.backward()
                self.optimizer.step()

    def increment_sample_size(self, feature, label):
        self.features[self.current_sample] = feature
        self.labels[self.current_sample] = label.astype(np.float64)
        self.current_sample += 1
        if self.current_sample >= self.num_samples:
            self.update_model()
            self.current_sample = 0

    def save_stats(self, games_played, games_won, moves_made, cells_left, guesses_made):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-5]    # 修改时间格式以精确到0.1秒
        with open(self.stats_file, "a") as f:
            record = f"{games_played};{games_won};{moves_made};{cells_left};{guesses_made};{timestamp};\n"
            f.write(record)

    def save_model(self, file_name):
        file_path = os.path.join(self.model_dir, file_name)
        torch.save(self.model.state_dict(), file_path)


if __name__ == "__main__":
    # python minesweeper_train_cnn.py --cuda --reward 30.0 --win-rate 0.99
    # tensorboard --logdir=./
    parser = argparse.ArgumentParser(description="Train a CNN to play Minesweeper")
    parser.add_argument("--cuda", default=False, action="store_true",
                        help="Enable cuda")
    parser.add_argument("--env", default=ENV_NAME,
                        help="Name of the environment, default=" + ENV_NAME)
    parser.add_argument("--reward", type=float, default=REWARD_BOUND,
                        help="Mean reward boundary for stop of training, default=%.2f" % REWARD_BOUND)
    parser.add_argument("--win-rate", type=float, default=WIN_RATE_BOUND,
                        help="Win rate boundary for stop of training, default=%.2f" % WIN_RATE_BOUND)
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = MinesweeperEnv(num_rows=6, num_cols=6, num_mines=4, render_mode="rgb_array")
    env = CustomObservationWrapper(env, encoding_type='one_hot')
    print(env)

    net = models.Model(env.observation_space.shape, env.action_space.n).to(device)
    writer = SummaryWriter(comment="-" + args.env)
    # 加载模型(在已有模型基础上继续训练)
    # model_path = "Minesweeper-v0-best_reward.dat"
    # net.load_state_dict(torch.load(model_path, map_location=device))
    print(net)

    agent = Agent(env, net, device, model_dir)

    list_episode_reward = []
    list_moves_made = []
    list_cells_left = []
    list_guesses_made = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_mean_episode_reward = None
    best_win_rate = None

    while True:
        frame_idx += 1

        episode_reward, moves_made, cells_left, guesses_made, games_played, games_won, stats = \
            agent.play_step()

        if episode_reward is not None:
            list_episode_reward.append(episode_reward)
            list_moves_made.append(moves_made)
            list_cells_left.append(cells_left)
            list_guesses_made.append(guesses_made)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            # 每一局显示统计信息
            # print("%d: done %d games, episode reward %.3f, moves made %d, cells left %d, speed %.2f f/s" %
            #       (frame_idx, games_played, episode_reward, moves_made, int(cells_left), speed))
            # print(f"episode stats: ", stats)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("episode_reward", episode_reward, frame_idx)
            writer.add_scalar("moves_made", moves_made, frame_idx)
            writer.add_scalar("cells_left", cells_left, frame_idx)

            if games_played % AGG_STATS_EVERY == 0:
                mean_episode_reward = np.mean(list_episode_reward)
                mean_moves_made = np.mean(list_moves_made)
                mean_cells_left = np.mean(list_cells_left)
                mean_guessed_made = np.mean(list_guesses_made)
                win_rate = games_won / AGG_STATS_EVERY
                writer.add_scalar("mean_episode_reward", mean_episode_reward, frame_idx)
                writer.add_scalar("win_rate", win_rate, frame_idx)
                # 保存统计信息到文件
                agent.save_stats(games_played, games_won, float(mean_moves_made), float(mean_cells_left),
                                 float(mean_guessed_made))
                print("%d: done %d games, mean episode reward %.3f, win rate %.2f" %
                      (frame_idx, games_played, float(mean_episode_reward), win_rate))
                if best_mean_episode_reward is None or best_mean_episode_reward < mean_episode_reward:
                    agent.save_model(args.env + "-best_reward.dat")
                    if best_mean_episode_reward is not None:
                        print("Best mean reward updated %.3f -> %.3f, model saved"
                              % (float(best_mean_episode_reward), float(mean_episode_reward)))
                    best_mean_episode_reward = mean_episode_reward
                if best_win_rate is None or best_win_rate < win_rate:
                    agent.save_model(args.env + "-best_win_rate.dat")
                    if best_win_rate is not None:
                        print("Best win rate updated %.3f -> %.3f, model saved"
                              % (float(best_win_rate), float(win_rate)))
                    best_win_rate = win_rate
                if mean_episode_reward >= args.reward and win_rate >= args.win_rate:
                    print("Solved in %d frames! Mean Episode reward: %.3f, Win rate: %.3f"
                          % (frame_idx, float(mean_episode_reward), float(win_rate)))
                    break
                agent.games_won = 0
                list_episode_reward.clear()
                list_moves_made.clear()
                list_cells_left.clear()
                list_guesses_made.clear()
            # 定期保存模型
            if games_played % SAVE_MODEL_EVERY == 0:
                checkpoint_path = f"{args.env}-checkpoint-{games_played}.dat"
                agent.save_model(checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")

    writer.close()
    os.system("shutdown")
