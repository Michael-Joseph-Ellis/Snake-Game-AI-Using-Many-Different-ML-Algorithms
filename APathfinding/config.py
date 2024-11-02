import pygame
import sys
import random
import heapq
import multiprocessing
import time
from pygame.locals import *

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (50, 50, 50)
SNAKECOLOR = (167, 187, 199)
HEADCOLOR = (41, 120, 181)
FOODCOLOR = (218, 127, 143)
BGCOLOR = (225, 229, 234)
SCREENCOLOR = (250, 243, 243)

FPS = 100

WINDOW_WIDTH = 520
WINDOW_HEIGHT = 600
SCREEN_SIZE = 500

GRID_SIZE = 20
GRID_WIDTH = SCREEN_SIZE // GRID_SIZE
GRID_HEIGHT = SCREEN_SIZE // GRID_SIZE

MARGIN = 10
TOP_MARGIN = 90

UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)