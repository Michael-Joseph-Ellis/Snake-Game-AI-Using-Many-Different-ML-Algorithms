import torch
import random
import numpy as np
import plotter
from collections import deque
from snakeGame import SnakeGame, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, EvoltuionarySnakeAI
import os 

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = EvoltuionarySnakeAI(11, 256, 3, self.model)  # Initialize the trainer
        
        # load model if it exists
        if os.path.exists('./model/model.pth'):
            self.model.load_state_dict(torch.load('./model/model.pth'))

    def get_state(self, game):
        head = game.head
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            game.is_collision() if dir_r else game.is_collision() if dir_l else game.is_collision() if dir_u else game.is_collision(),

            # Danger right
            game.is_collision() if dir_r else game.is_collision() if dir_l else game.is_collision() if dir_u else game.is_collision(),

            # Danger left
            game.is_collision() if dir_r else game.is_collision() if dir_l else game.is_collision() if dir_u else game.is_collision(),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < head.x,  # food left
            game.food.x > head.x,  # food right
            game.food.y < head.y,  # food up
            game.food.y > head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, game):
        """
        Get the action to take based on the current state.

        Args:
            state (np.array): The current state representation.
            game (SnakeGame): The SnakeGame instance.

        Returns:
            list: The one-hot encoded action to take.
        """
        # Penalize if snake collides with itself or the wall
        if game.crash:
            return [-2, 0, 0]

        # Trade-off exploration/exploitation
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            action = [0, 0, 0]
            action[move] = 1
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()
            action = [0, 0, 0]
            action[move] = 1

        return action

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGame()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old, game)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()
                agent.high_score = record

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plotter.plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
