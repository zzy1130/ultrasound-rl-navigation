import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class DQN(nn.Module):
    def __init__(self, input_channels=1, num_actions=5):
        super(DQN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_out_size(input_channels, 64, 64)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
    
    def _get_conv_out_size(self, input_channels, h, w):
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, h, w)
            conv_out = self.conv_layers(dummy_input)
            return int(np.prod(conv_out.size()))
    
    def forward(self, x):
        conv_out = self.conv_layers(x)
        conv_out = conv_out.view(conv_out.size(0), -1)
        return self.fc_layers(conv_out)


class LegacyDQN(nn.Module):
    """
    Legacy architecture to load checkpoints that store keys like
    features.* and fc.* (as in results/trained_models/agent_final.pt).
    Two conv layers with stride 2 to downsample 64x64 -> 16x16.
    Conv modules are positioned at indices 0 and 3 to match checkpoint keys.
    """

    def __init__(self, input_channels=1, num_actions=5):
        super(LegacyDQN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ReLU(inplace=True),  # placeholder to shift index
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 16 * 16, 256),  # fc.0
            nn.ReLU(inplace=True),
            nn.Linear(256, num_actions),   # fc.2
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class NavigationAgent:
    def __init__(self, num_actions, learning_rate=0.0005, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
                 batch_size=64, target_update=10, device='cpu'):
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = device
        
        self.q_network = DQN(num_actions=num_actions).to(device)
        self.target_network = DQN(num_actions=num_actions).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        self.memory = deque(maxlen=10000)
        self.steps_done = 0
        
        self.update_target_network()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True, stochastic_policy=False, temperature=1.0):
        # Epsilon exploration (training only)
        if training and random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor).squeeze(0)

            if stochastic_policy:
                # Softmax sampling for stochastic policy
                probs = torch.softmax(q_values / max(1e-6, temperature), dim=0)
                return torch.multinomial(probs, 1).item()

            # Deterministic argmax
            return q_values.max(0)[1].item()
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0.0
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.update_target_network()
        
        return loss.item()
    
    def save(self, filepath):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, filepath)
    
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        # New-format checkpoint
        if 'q_network_state_dict' in checkpoint:
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.steps_done = checkpoint.get('steps_done', self.steps_done)
        # Legacy checkpoint with policy_net/target_net keys
        elif 'policy_net' in checkpoint and 'target_net' in checkpoint:
            legacy_q = LegacyDQN(num_actions=self.num_actions).to(self.device)
            legacy_target = LegacyDQN(num_actions=self.num_actions).to(self.device)
            legacy_q.load_state_dict(checkpoint['policy_net'])
            legacy_target.load_state_dict(checkpoint['target_net'])
            self.q_network = legacy_q
            self.target_network = legacy_target
            # Reset optimizer to match new parameter set
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.optimizer.param_groups[0]['lr'])
            # Optimizer state may not match; skip to avoid size errors
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.steps_done = checkpoint.get('episode_count', self.steps_done)
        else:
            raise KeyError("Unrecognized checkpoint format")
