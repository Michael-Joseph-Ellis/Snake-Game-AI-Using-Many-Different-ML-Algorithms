import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
COLORS = {
    "WHITE": (255, 255, 255),
    "RED": (200,0,0),
    "BLUE1": (0, 0, 255),
    "BLUE2": (0, 100, 255),
    "BLACK": (0,0,0),
    "PINK1": (255,200,200),
    "GREEN": (0,255,0),
}

BLOCK_SIZE = 20
SPEED = 40

class SnakeGameAI:
    """
    Snake Game AI class.
    """

    def __init__(self, w=640, h=480):
        """
        Initialize the Snake Game AI.

        Args:
            w (int): Width of the game window. Default is 640.
            h (int): Height of the game window. Default is 480.
        """
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h), pygame.HWSURFACE)
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def __str__(self):
        return f'SnakeGameAI(w={self.w}, h={self.h}, score={self.score})'

    def reset(self):
        """
        Reset the game state.
        """
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.crash = False # reset the crash attribute to false 

    def _place_food(self):
        """
        Place the food randomly on the game grid.
        """
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        """
        Play a step of the game.

        Args:
            action (list): The action to take. It should be a one-hot encoded list representing the direction.

        Returns:
            tuple: A tuple containing the reward, game over flag, and current score.
        """
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            
            elif event.type == pygame.MOUSEBUTTONUP:
                pass
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pass
            elif event.type == pygame.MOUSEMOTION:
                pass

        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            self.crash = True
            game_over = True
            reward = -2
            return reward, game_over, self.score
        
        # calculate the distance to the food 
        food_distance_before = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 3
            self._place_food()
        else:
            # calculate the distance to the food after the move 
            food_distance_after = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
            # if the snake gets closer to the food, provide a positive reward 
            if food_distance_after < food_distance_before:
                reward = 3
            else:
                reward = -2
                
            self.snake.pop()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        """
        Check if there is a collision with the boundaries or the snake itself.

        Args:
            pt (Point): The point to check for collision. If not provided, the head of the snake is used.

        Returns:
            bool: True if there is a collision, False otherwise.
        """
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        """
        Update the game UI.
        """
        self.display.fill(COLORS["BLACK"])

        for pt in self.snake:
            pygame.draw.rect(self.display, COLORS["PINK1"], pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, COLORS["PINK1"], pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, COLORS["GREEN"], pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, COLORS["WHITE"])
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self):
         
        # Calculate the Euclidean distance between the snake's head and the fruit
        distance_x = self.food.x - self.head.x
        distance_y = self.food.y - self.head.y
        
        # Choose the direction that minimizes the distance
        if abs(distance_x) > abs(distance_y):
            if distance_x > 0 and self.direction != Direction.LEFT:
                self.direction = Direction.RIGHT
            elif distance_x < 0 and self.direction != Direction.RIGHT:
                self.direction = Direction.LEFT
        else:
            if distance_y > 0 and self.direction != Direction.UP:
                self.direction = Direction.DOWN
            elif distance_y < 0 and self.direction != Direction.DOWN:
                self.direction = Direction.UP
        
        # Update the snake's head position based on the chosen direction
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