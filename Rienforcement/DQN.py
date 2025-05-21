# DQN.py
import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, models
 
class DQNAgent:
    def __init__(
        self, 
        state_size,
        action_size,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        batch_size=64,
        memory_size=10000,
        target_update_freq=5  # update target network every 5 episodes
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.target_update_freq = target_update_freq
        self.train_step = 0  # count training episodes for target update
        
        # Main model and target model
        self.model = self._build_model(learning_rate)
        self.target_model = self._build_model(learning_rate)
        self.update_target_model()
    
    def _build_model(self, lr):
        """Build a deep MLP model for Q-value approximation."""
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_dim=self.state_size),
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(
            loss=tf.keras.losses.Huber(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
        )
        return model
    
    def update_target_model(self):
        """Copy weights from the main model to the target model."""
        self.target_model.set_weights(self.model.get_weights())

    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """
        Epsilon-greedy action selection.
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state[np.newaxis, :], verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self):
        """
        Sample a batch and train the network using the target network.
        """
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([experience[0] for experience in batch])
        actions = np.array([experience[1] for experience in batch])
        rewards = np.array([experience[2] for experience in batch])
        next_states = np.array([experience[3] for experience in batch])
        dones = np.array([experience[4] for experience in batch])
        
        # Predict Q-values for current states and next states
        q_current = self.model.predict(states, verbose=0)
        q_next = self.target_model.predict(next_states, verbose=0)
        
        for i in range(self.batch_size):
            if dones[i]:
                q_current[i][actions[i]] = rewards[i]
            else:
                q_current[i][actions[i]] = rewards[i] + self.gamma * np.max(q_next[i])
        
        # Train on the batch
        self.model.fit(states, q_current, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
