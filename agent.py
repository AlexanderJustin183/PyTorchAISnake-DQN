#!usr/bin/env python
# -*- coding: UTF-8 -*-

__author__ = "AlexanderJustin"
# My blog: https://www.blog.csdn.net/AlexanderJustin183/article
__version__ = "3.0.1"

__doc__ = """
Agent Program for Snake AI.
It has the ``main`` function.
If you want to run this program, you should execute this command:
    $ pip install -r requirements.txt
Date: 2022-11-22
"""

import sys
from configparser import ConfigParser
import random
from collections import deque

import numpy as np
import torch

from game import Direction, Point, SnakeGameAI
from helper import *
from model import Linear_QNet, QTrainer


class NoCudaError(Exception):
    """docstring for NoCudaError"""
    
    def __init__(self, msg):
        self.msg = msg
        
    def __str__(self):
        return self.msg


class Agent:
    """docstring for Agent"""
    
    def __init__(self, lr, max_memory, bt, use_cuda=False):
        self.lr = lr
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=max_memory)  # popleft()
        
        # self.hidden_size = (2 ** 8) * 1
        # self.model = Linear_QNet1(11, self.hidden_size, use_cuda=use_cuda)
        self.hidden_size = (2 ** 9) * 1
        self.model = Linear_QNet(11, self.hidden_size, use_cuda=use_cuda)
        # self.model.load_state_dict(torch.load("./model/model.pth"))
        self.cuda = use_cuda
        
        self.batch_size = bt
        self.game = None
        self.trainer = QTrainer(self.model, lr=lr, gamma=self.gamma, use_cuda=use_cuda)
    
    def get_state(self, game):
        self.game = game
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        
        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),
            
            # Danger Right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),
            
            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_r,
            dir_l,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y,  # food down
        
        ]
        
        return np.array(state, dtype=int)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append(
            (state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached
    
    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)  # list of tuples
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        # for state, action, reward, next_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)
    
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_action(self, state, max_games=70):
        # random moves: tradeoff exploration / exploitation
        
        self.epsilon = max_games - self.n_games
        # head = self.game.snake[0]
        final_move = [0, 0, 0]
        # if random.randint(0, 200) < 40 and self.n_games < 80:
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
        else:
            
            state0 = torch.tensor(state, dtype=torch.float)
            if self.cuda:
                state0 = state0.cuda()
            prediction = self.model(state0)
            move = torch.argmax(prediction).item() % 3
        final_move[move] = 1
        
        return final_move


def read_config(filename, sections, keys):
    """read a config value."""
    cfg = ConfigParser()
    cfg.read(filename)
    setting = []
    for i in sections:
        for j in keys:
            try:
                setting.append(str(cfg[i][j]))
            except KeyError:
                print("[ERROR CODE 1] Invalid settings format.")
                sys.exit()
    return setting


def train(ini_file="settings.ini"):
    # print(torch.cuda.is_available())
    cfg = read_config(ini_file,
                      ["Settings"], ["max_games", "speed", "LR", "MAX_MEMORY", "BATCH_SIZE", "use_cuda"])
    n = int(cfg[0]) if cfg[0].upper() != "INF" else float(np.inf)
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    cuda = cfg[5].lower() == "true"
    
    if not torch.cuda.is_available():
        raise NoCudaError(
            """Sorry, you haven't a torch with cuda. You can set cuda to false,
            or install torch with cuda:
            CONDA.EXE install torch==1.13.0+cu117 torchvision torchaudio
            pip install torch==1.13.0+cu117 torchvision torchaudio
            """
        )
    
    agent = Agent(lr=float(cfg[2]), bt=int(cfg[4]), max_memory=int(cfg[3]), use_cuda=cuda)
    
    game = SnakeGameAI(speed=int(cfg[1]), use_cuda=cuda)
    game.final = n
    game.lr = agent.lr
    while agent.n_games <= n:
        game.epsilon = agent.epsilon  # get old state
        state_old = agent.get_state(game)  # get move
        final_move = agent.get_action(state_old)  # , game.final)
        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)  # train short memory
        # remember
        agent.remember(state_old, final_move, reward, state_new, done)
        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            
            if score > record:
                record = score
                game.rec = record
                print("=========== Game", agent.n_games, "crushed the record ==============")
                agent.model.save()
            
            game.game = agent.n_games
            print("Game", agent.n_games, "Score", score, "Record:", record)
            
            plot_scores.append(score)
            total_score += score
            mean_score = (lambda x, n: float(("%%.%df" % n) % x))((total_score / agent.n_games), 3)
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores, game.w)
    if not os.path.exists("./records"):
        os.makedirs("./records")
    print(str(record), file=open("records/rec_%d.txt" % n, "w"))
    save(n, cuda)


def main():
    """ main function """
    if len(sys.argv) >= 2:
        train(sys.argv[1])
    else:
        train()


if __name__ == "__main__":
    main()
