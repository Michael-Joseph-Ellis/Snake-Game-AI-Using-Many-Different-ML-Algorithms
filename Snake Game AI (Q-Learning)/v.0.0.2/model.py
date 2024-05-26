import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
import copy
import snakeGame

class Linear_QNet(nn.Module):
    """
    A linear Q-network model for reinforcement learning.

    Args:
        input_size (int): The size of the input state.
        hidden_size (int): The size of the hidden layer.
        output_size (int): The size of the output action space.
    """
    
    input_size = 10
    output_size = 5
    
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): The input state tensor.

        Returns:
            torch.Tensor: The output Q-values tensor.
        """
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        """
        Save the model's state dictionary to a file.

        Args:
            file_name (str, optional): The name of the file to save the model. Defaults to 'model.pth'.
        """
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
        
class EvoltuionarySnakeAI:
    def __init__(self, population_size, input_size, hidden_size, output_size):
        self.population_size = population_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.population = [Linear_QNet(input_size, hidden_size, output_size) for _ in range(population_size)]
        
    def evolve(self):
        # Evaluate fitness
        fitness_scores = [self.evaluate(individual) for individual in self.population]
        
        # Selection 
        selected_parents = self.select_parents(fitness_scores)
        
        # Crossover and mutation 
        offspring = self.crossover_and_mutate(selected_parents)
        
        # Replace population 
        self.population = offspring
    
    def evaluate(self, individual):
        total_score = 0
        num_episodes = 10  # Number of episodes (games) to play
        for _ in range(num_episodes):
            # Initialize the Snake game environment
            game = snakeGame.SnakeGame()
            done = False
            while not done:
                # Get the current state of the game and convert it to input tensor
                state = game.get_state()
                state_tensor = torch.tensor(state, dtype=torch.float32)

                # Pass the state through the individual (neural network) to get Q-values
                with torch.no_grad():
                    q_values = individual(state_tensor)

                # Choose action with highest Q-value
                action = torch.argmax(q_values).item()

                # Perform action and get next state and reward
                next_state, reward, done = game.play(action)

                # Accumulate total score
                total_score += reward

        # Calculate average score over all episodes
        average_score = total_score / num_episodes

        return average_score
    
    def select_parents(self, fitness_scores):
        # Convert fitness scores to selection probabilities
        selection_probs = np.array(fitness_scores) / sum(fitness_scores)

        # Select parents based on their selection probabilities
        selected_indices = np.random.choice(len(self.population), size=self.population_size, replace=True, p=selection_probs)

        # Retrieve selected parents
        selected_parents = [self.population[i] for i in selected_indices]

        return selected_parents
    
    def crossover_and_mutate(self, selected_parents):
        offspring = []
        for _ in range(self.population_size):
            # Randomly select two parents for crossover
            parent1, parent2 = np.random.choice(selected_parents, size=2, replace=False)

            # Create a copy of one of the parents as the initial offspring
            child = copy.deepcopy(parent1)

            # Perform crossover
            crossover_point = np.random.randint(0, len(parent1.parameters()))  # Choose a random crossover point
            for i, (param_name, param) in enumerate(child.named_parameters()):
                if i < crossover_point:
                    # Take parameter from parent 1
                    param.data.copy_(parent1.state_dict()[param_name])
                else:
                    # Take parameter from parent 2
                    param.data.copy_(parent2.state_dict()[param_name])

            # Perform mutation
            mutation_rate = 0.1  # Adjust as needed
            for param in child.parameters():
                if np.random.rand() < mutation_rate:
                    # Add random noise to the parameter
                    param.data += torch.randn_like(param.data) * 0.1  # Adjust mutation magnitude as needed

            # Add the offspring to the list
            offspring.append(child)

        return offspring