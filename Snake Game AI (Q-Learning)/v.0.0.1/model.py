import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    """
    A linear Q-network model for reinforcement learning.

    Args:
        input_size (int): The size of the input state.
        hidden_size (int): The size of the hidden layer.
        output_size (int): The size of the output action space.
    """
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


class QTrainer:
    """
    A trainer class for training the Q-network model.

    Args:
        model (nn.Module): The Q-network model.
        lr (float): The learning rate for the optimizer.
        gamma (float): The discount factor for future rewards.
    """
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
    
    def train_step(self, state, action, reward, next_state, done):
        """
        Perform a single training step.

        Args:
            state (list or np.array): The current state.
            action (int): The action taken in the current state.
            reward (float): The reward received for taking the action.
            next_state (list or np.array): The next state after taking the action.
            done (bool): Whether the episode is done or not.
        """
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()

        """
        Summary: 
        1: predicted Q values with current state
        2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        3: preds[argmax(action)] = Q_new
        4: loss = mse(preds, target)
        5: backpropogation
        6: update weights
        7: repeat
        """