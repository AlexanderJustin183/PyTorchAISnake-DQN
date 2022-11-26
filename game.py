#!usr/bin/env python
# -*- coding: UTF-8 -*-

__author__ = "AlexanderJustin"
# My blog: https://www.blog.csdn.net/AlexanderJustin183/article
__version__ = "3.0.1"

__doc__ = """
Game graphics module for Snake AI.
Date: 2022-11-22
"""

import os
import random
from collections import namedtuple
from enum import Enum

import numpy as np
import pygame
from pygame import Color

from helper import save

# Pygame: https://www.pygame.org/
# use command `` pip install pygame --user `` to install.

pygame.init()
font = pygame.font.SysFont("Arial", 22)
title_font = pygame.font.Font("free_sans_bold.ttf", 100)
msg_font = pygame.font.Font("free_sans_bold.ttf", 20)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple("Point", "x, y")

# RGB colors
WHITE = Color("white")
RED = Color("0xff00ff")
GREEN1 = Color("0x00ff00")
BLACK = Color("black")

GREEN2 = Color("0x00b400")
BLUE = Color("0x00d0f0")
GREEN3 = Color("0x00c800")


BLOCK_SIZE = 20
GAP_SIZE = 2
SMALL_BLOCK_SIZE = BLOCK_SIZE - GAP_SIZE - 1

# SPEED = 40
SPEED = 4000


class SnakeGameAI:
    """docstring for SnakeGameAI"""
    
    def __init__(self, w=640, h=480, speed=SPEED, use_cuda=False):
        self.w = w
        self.h = h
        self.speed = speed
        # init display
        os.environ["SDL_VIDEO_WINDOW_POS"] = "%d,%d" % (0, 30)
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snake AI with DQN. Copyright AlexanderJustin © 2022")
        self.direction = Direction.RIGHT
        self.head = None
        self.score = 0
        self.food = None
        self.snake = []
        self.frame_iteration = 0
        self.clock = pygame.time.Clock()
        self.game = 1
        self.rec = 0
        self.lr = 0
        self.final = 1
        self.epsilon = 0.00
        # self.start()
        self.cuda = use_cuda
        self.reset()
    
    def terminate(self):
        save(self.final, self.cuda)
        pygame.quit()
        quit()
    
    def start(self):
        
        title_content = title_font.render("Snake!", True, WHITE, GREEN1)
        angle = 0
        while True:
            self.display.fill(BLACK)
            # title
            rotated_title = pygame.transform.rotate(title_content, angle)
            rotated_title_rect = rotated_title.get_rect()
            rotated_title_rect.center = (self.w / 2, self.h / 2)
            self.display.blit(rotated_title, rotated_title_rect)
            # message
            msg = msg_font.render("Press any key to play.", True, WHITE)
            msg_rect = msg.get_rect()
            msg_rect.bottomright = (self.w - 10, self.h - 15)
            self.display.blit(msg, msg_rect)
            # check for event
            event = pygame.event.poll()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.terminate()
                else:
                    return None
            elif event.type == pygame.QUIT:
                self.terminate()
            pygame.display.update()
            pygame.time.delay(30)
            angle += 3
    
    def reset(self):
        # init game state
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        
        self._place_food()
        self.frame_iteration = 0
    
    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
    
    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.terminate()
        
        # 2. move
        self._move(action)  # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        
        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        # pygame.display.set_caption("Snake running on %d FPS" % self.clock.get_fps())
        self.clock.tick(self.speed)
        # 6. return game over and score
        return reward, game_over, self.score
    
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        
        return False
    
    def draw_info(self):
        text = font.render("CURRENT GAME NUMBER: " + str(self.game), True, WHITE)
        self.display.blit(text, [10, 10])
        text = font.render("SCORE: " + str(self.score), True, WHITE)
        self.display.blit(text, [10, 35])
        text = font.render("EPSILON: " + str(self.epsilon), True, WHITE)
        self.display.blit(text, [10, 60])
        text = font.render("LEARNING RATE: %g" % self.lr, True, WHITE)
        self.display.blit(text, [10, 85])
        text = font.render("RECORD: " + str(self.rec), True, WHITE)
        self.display.blit(text, [10, 110])
        text = font.render("" if self.epsilon < 0 else "Exploring...", True, BLUE)
        self.display.blit(text, [10, 135])
    
    def block(self, color, x1, y1, x2, y2):
        pygame.draw.polygon(self.display, color,
                            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    
    def _update_ui(self):
        self.display.fill(BLACK)
        # temp = self.snake[0]
        for pt in self.snake:
            # pygame.draw.rect(self.display, GREEN3,
            #                  pygame.Rect(pt.x + GAP_SIZE,
            #                              pt.y + GAP_SIZE,
            #                              BLOCK_SIZE - GAP_SIZE, BLOCK_SIZE - GAP_SIZE))
            #
            # ix = pt.x + GAP_SIZE
            # iy = pt.y + GAP_SIZE
            # tx = temp.x + GAP_SIZE
            # ty = temp.y + GAP_SIZE
            #
            # if temp.x == pt.x:
            #     if pt.y > temp.y:
            #         self.block(GREEN3, tx, ty + SMALL_BLOCK_SIZE, ix + SMALL_BLOCK_SIZE, iy)
            #     if pt.y < temp.y:
            #         self.block(GREEN3, ix, iy + SMALL_BLOCK_SIZE, tx + SMALL_BLOCK_SIZE, ty)
            # if temp.y == pt.y:
            #     if pt.x > temp.x:
            #         self.block(GREEN3, tx + SMALL_BLOCK_SIZE, ty, ix, iy + SMALL_BLOCK_SIZE)
            #     if pt.x < temp.x:
            #         self.block(GREEN3, ix + SMALL_BLOCK_SIZE, iy, tx, ty + SMALL_BLOCK_SIZE)
            # temp = pt
            # """
            pygame.draw.rect(self.display, GREEN1,
                             pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, GREEN2,
                             pygame.Rect(pt.x + 4, pt.y + 4, BLOCK_SIZE - 8, BLOCK_SIZE - 8))
            # """
        
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x + GAP_SIZE,
                                                        self.food.y + GAP_SIZE, SMALL_BLOCK_SIZE, SMALL_BLOCK_SIZE))
        
        self.draw_info()
        pygame.display.update()
    
    def _move(self, action):
        # [straight, right, left]
        
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % len(clock_wise)
            new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % len(clock_wise)
            new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d
        
        self.direction = new_dir
        
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        
        self.head = Point(x, y)

#
# app.io.funcIO(async.Batch().func_base(myWrapper))
