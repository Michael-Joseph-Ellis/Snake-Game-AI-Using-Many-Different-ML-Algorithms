# Snake Game with A* Pathfinding Algorithm

This project implements the classic Snake game, enhanced with an A* pathfinding algorithm to train an AI agent to navigate the game. The AI leverages heuristic-based search to determine the optimal path to the food, aiming to maximize the score while avoiding collisions.

## Overview
The Snake game involves maneuvering a line (the snake), which grows in length with each piece of food it consumes. The game ends when the snake collides with itself or the game boundaries. This implementation uses the A* pathfinding algorithm to play the game autonomously.

## Algorithm Details

A* is a popular pathfinding and graph traversal algorithm often used in games and robotics. It combines the benefits of Dijkstra's algorithm and a greedy best-first search, using a heuristic to prioritize paths.

### Key Concepts:

- **Node**: Represents a state in the game, including the position of the snake and the game board state.
- **Open List**: A priority queue of nodes that are yet to be evaluated.
- **Closed List**: A list of nodes that have already been evaluated.
- **Heuristic (h)**: An estimated cost from the current node to the goal, calculated using the Manhattan distance.
- **Cost (g)**: The cost from the start node to the current node.
- **Total Cost (f)**: The sum of the cost `g` and the heuristic `h`. (f = g + h).

### A* Search Update Rule:

The algorithm evaluates nodes based on their total cost `f`. Nodes with the lowest `f` values are expanded first. The goal is to find the shortest path from the snake's head to the food while avoiding collisions.

## AI Mechanics

The Snake AI is implemented using the A* pathfinding algorithm:

- The game board is modeled as a grid, with each cell representing a possible position for the snake.
- The AI calculates the shortest path to the food using the heuristic-based search, avoiding obstacles and the snake's body.
- The AI dynamically updates its path as the snake and food positions change.

## Key Functions:

- **Node Class**: Manages information about each state, including position, cost, and parent for backtracking.
- **SnakeAI Class**: Implements the A* algorithm, handling the game board setup, heuristic calculations, and pathfinding.
- **Graph Visualization**: A separate process visualizes the number of apples collected per game, using `matplotlib`.

## Results
The A* pathfinding algorithm provides a more deterministic and efficient way for the AI to navigate the game than other algorithms or models that learn and adapt over time. The AI efficiently finds paths to the food using heuristic-based decisions, making quick and optimal moves in many situations. However, A* does not improve or adapt from past experiences. Its strategy remains static, leading to inconsistent performance in more complex game scenarios. The AI may struggle with obstacles or fail to handle unexpected situations as effectively as adaptive or learning-based approaches.

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.