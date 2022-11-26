### `Teach AI To Play Snake! Reinforcement Learning With PyTorch and Pygame`

In this Python Reinforcement Learning Tutorial series we teach an AI to play Snake!
We build everything from scratch using Pygame and PyTorch. The tutorial consists of 4 parts:

You can find all tutorials on my
channel: [Playlist](https://www.youtube.com/playlist?list=PLqnslRFeH2UrDh7vUmJ60YrmWd64mTTKV)

- Part 1: I'll show you the project and teach you some basics about Reinforcement Learning and Deep Q Learning.
- Part 2: Learn how to ```set up the environment``` and implement the Snake game.
- Part 3: Implement the``` agent``` that controls the game.
- Part 4: Implement the ```neural network``` to predict the moves and train it.

### `Some Code(part)`

agent.py

```python
from collections import deque

from game import Direction, Point
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
        self.hidden_size = (2 ** 8) * 10
        # hidden_size = (2 ** 8) * 1
        # print(hidden_size)
        self.model = Linear_QNet(11, self.hidden_size, use_cuda=use_cuda)
        self.cuda = use_cuda
      
        self.model = self.model
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
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)) or
            (dir_l and game.is_collision(point_l)),
          
            # Danger right
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
            dir_l,
            dir_r,
            dir_u,
            dir_d,
          
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y,  # food down
      
        ]
        return state

```

game.py

```python
import random
from collections import namedtuple
from enum import Enum

import pygame

pygame.init()

font = pygame.font.SysFont("arial", 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple("Point", "x, y")

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (255, 0, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 10


class SnakeGame:
  
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()
      
        # init game state
        self.direction = Direction.RIGHT
      
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
      
        self.score = 0
        self.food = None
        self._place_food()
  
    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

```

### `Package installation`

- torch: ```pip install torch==1.13.1+cu117 torchvision torchaudio```or [torch](https://www.pytorch.org)
- pygame: ```pip install pygame --user```or [pygame](https://www.pygame.org)
- matplotlib: ```pip install matplotlib==3.5.0```or [matplotlib](https://matplotlib.org/3.5.0/index.html#installation)
