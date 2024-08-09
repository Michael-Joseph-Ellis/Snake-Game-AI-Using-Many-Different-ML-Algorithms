# Snake Game with Q-Learning

This project implements the classic Snake game, enhanced with a Q-learning algorithm to train an AI agent to navigate the game. The AI utilizes reinforcement learning to adjust its strategies based on received rewards and penalties, aiming to maximize its score by learning optimal actions.

## Overview

The Snake game involves maneuvering a line (the snake), which grows in length with each piece of food it consumes. The game ends when the snake collides with itself or the game boundaries. This implementation uses a Q-learning algorithm to play the game autonomously.

## Algorithm Details

Q-learning is a model-free reinforcement learning algorithm that aims to find the best action for any given state. It does this by learning a Q-value function, which predicts the expected rewards of action-state pairs.

### Key Concepts:

- **State**: The current configuration of the game, including the snake's position and the food's location.
- **Action**: Possible directions for the snake’s movement (up, down, left, right).
- **Reward**: Feedback from the environment, such as points for eating food or penalties for collisions.
- **Q-value**: An estimation of future rewards for actions taken in a specific state.

### Q-learning Update Rule:
Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]

where:
- `s` is the current state,
- `a` is the action taken,
- `r` is the reward received,
- `s'` is the subsequent state,
- `α'` represents all possible next actions from state `s'`,
- `α` is the learning rate,
- `γ` is the discount factor.

## Training the Agent

The training involves the AI agent engaging in numerous game sessions, exploring various strategies, and refining its decision-making process based on the game outcomes. The learning curve is gradual, with significant time required for noticeable improvements.

## Results

The performance of the AI agent after training is mediocre. The algorithm performs at a level similar to that of an average or casual human player. Progression and improvement in the AI's gameplay occur slowly, and the agent requires considerable time to enhance its performance significantly.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
