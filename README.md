# Snake Game AI Using Many Different ML Algorithms

This project implements the classic Snake game, featuring AI agents trained with various machine learning algorithms to master the game. The goal is to explore how different AI approaches perform in a controlled environment, with Snake as the testbed.

## Table of Contents

- [Snake Game AI Using Many Different ML Algorithms](#snake-game-ai-using-many-different-ml-algorithms)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Features](#features)
  - [Algorithms Implemented](#algorithms-implemented)
  - [License](#license)

## Introduction

This repository contains the code for a Snake game AI where the snake learns to play using different machine learning algorithms. The primary aim of this project is to compare the effectiveness of various algorithms in a simple yet challenging environment.

## Features

- **Multiple AI Agents:** Implemented using different machine learning algorithms.
- **Customizable Game Environment:** Modify the game grid size and speed.
- **Performance Tracking:** Compare the performance of different algorithms based on the score and longevity of the snake.

## Algorithms Implemented

1. **Greedy Algorithm** (Future Project)
   - The snake always moves towards the food using the shortest possible path. This can be done using basic pathfinding techniques like Breadth-First Search (BFS) or Depth-First Search (DFS).

2. **A* Pathfinding** (WIP)
   - A more sophisticated pathfinding algorithm that considers both the cost to reach the food and an estimated cost from the current position to the food, leading to more efficient routes.

3. **Hamiltonian Path** (Future Project)
   - Involves creating a path that visits every square on the grid exactly once, ensuring that the snake eventually reaches the food as it traverses the grid.

4. **Reinforcement Learning (e.g., Q-Learning, Deep Q-Networks)** (DONE)
   - The AI learns to play the game by being rewarded for reaching the food and penalized for collisions or dying, gradually improving its strategy.

5. **Genetic Algorithms** (Future Project)
   - Evolve a population of snakes over several generations, selecting for those that perform well, and improving performance through crossover and mutation.

6. **Minimax Algorithm with Alpha-Beta Pruning** (Future Project)
   - The snake simulates all possible moves and chooses the one that minimizes the worst-case scenario, ensuring optimal decision-making.

7. **Depth-First Search with Backtracking** (Future Project)
   - The snake explores all possible paths to the food and backtracks when it hits a dead-end, ensuring that it can reach the food without getting trapped.

8. **Flood Fill Algorithm** (Future Project)
   - The snake fills the grid from its position to the food and measures the area it can safely move to, helping it decide whether to move directly towards the food or take a detour.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
