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

class SnakeGame:
    """
    Snake Game class.
    """

    def __init__(self, w=640, h=480):
        """
        Initialize the Snake Game.

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
        return f'SnakeGame(w={self.w}, h={self.h}, score={self.score})'

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
        self._move(action)  # update the head
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


    def is_collision(self):
        """
        Check if there is a collision with the boundaries or the snake itself.

        Returns:
            bool: True if there is a collision, False otherwise.
        """
        # Check if the head of the snake collides with the boundaries
        if (
            self.head.x > self.w - BLOCK_SIZE or
            self.head.x < 0 or
            self.head.y > self.h - BLOCK_SIZE or
            self.head.y < 0
        ):
            return True
        
        # Check if the head of the snake collides with its own body
        if self.head in self.snake[1:]:
            return True

        return False

    def _move(self, action):
        """
        Move the snake based on the action.

        Args:
            action (list): The action to take. It should be a one-hot encoded list representing the direction.
        """
        # Decode the action
        if action[0] == 1:
            new_direction = Direction.RIGHT
        elif action[1] == 1:
            new_direction = Direction.LEFT
        elif action[2] == 1:
            new_direction = Direction.UP
        elif action[3] == 1:
            new_direction = Direction.DOWN
        
        # Update the direction if it's not opposite to the current direction
        if new_direction == Direction.RIGHT and self.direction != Direction.LEFT:
            self.direction = new_direction
        elif new_direction == Direction.LEFT and self.direction != Direction.RIGHT:
            self.direction = new_direction
        elif new_direction == Direction.UP and self.direction != Direction.DOWN:
            self.direction = new_direction
        elif new_direction == Direction.DOWN and self.direction != Direction.UP:
            self.direction = new_direction
        
        # Update the position of the snake's head based on the new direction
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

    def _update_ui(self):
        """
        Update the user interface (UI) to reflect the current state of the game.
        """
        self.display.fill(COLORS["BLACK"])  # Clear the display with a black background

        # Draw the snake
        for pt in self.snake:
            pygame.draw.rect(self.display, COLORS["GREEN"], pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))

        # Draw the food
        pygame.draw.rect(self.display, COLORS["RED"], pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # Draw the score
        text = font.render(f"Score: {self.score}", True, COLORS["WHITE"])
        self.display.blit(text, [0, 0])

        # Update the display
        pygame.display.flip()
