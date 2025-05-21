import gym
import numpy as np
import torch
from gym import spaces

class StockTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df, initial_balance=10000, trade_fraction=0.1, device=None):
        super(StockTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.n_step = len(self.df)
        self.initial_balance = initial_balance
        self.trade_fraction = trade_fraction
        self.balance = initial_balance
        self.owned_shares = 0.0  
        self.current_step = 0
        self.max_shares = 1000  
        
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        self.action_space = spaces.Discrete(2)
        
        obs_low = np.array([0, -1, 0, 0, 0, -1, -1, -1], dtype=np.float32)
        obs_high = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        
        self.transaction_cost = 0.005  # percentage cost per trade
        self.slippage = 0.002          # simulated slippage percentage
        self.reset()
    
    def _get_observation(self):
        time_progress = self.current_step / self.n_step
        current_price = self.df.loc[self.current_step, 'Close_Price']
        owned_norm = min(self.owned_shares / self.max_shares, 1.0)
        balance_norm = self.balance / self.initial_balance
        rsi = self.df.loc[self.current_step, 'RSI'] / 100.0  
        ma10 = self.df.loc[self.current_step, 'MA_10']
        ma50 = self.df.loc[self.current_step, 'MA_50']
   
        predicted_price = 0.6 * ma10 + 0.4 * ma50
        volume = self.df.loc[self.current_step, 'Volume']
        
        obs = np.array([time_progress, current_price, owned_norm, balance_norm, rsi, ma10, predicted_price, volume], dtype=np.float32)
        return torch.tensor(obs, device=self.device)
    
    def step(self, action):
        done = False
        current_price = self.df.loc[self.current_step, 'Close_Price']
   
        if action == 0:  # Buy
            executed_price = current_price * (1 + self.slippage)
        elif action == 1:  # Sell
            executed_price = current_price * (1 - self.slippage)
        
        old_portfolio_value = self.balance + self.owned_shares * current_price
        
        # Execute trade based on action
        if action == 0:  # Buy
            spend = self.trade_fraction * self.balance
            shares_to_buy = spend / executed_price
            if shares_to_buy > 0:
                self.owned_shares += shares_to_buy
                self.balance -= shares_to_buy * executed_price
        elif action == 1:  # Sell
            shares_to_sell = self.trade_fraction * self.owned_shares
            if shares_to_sell > 0:
                self.owned_shares -= shares_to_sell
                self.balance += shares_to_sell * executed_price
        
        self.current_step += 1
        if self.current_step >= self.n_step - 1:
            done = True
        
        next_price = self.df.loc[self.current_step, 'Close_Price'] if not done else current_price
        new_portfolio_value = self.balance + self.owned_shares * next_price
            
        base_reward = ((new_portfolio_value - old_portfolio_value) / (old_portfolio_value + 1e-7)) * 100.0 * 1.5

        drawdown_penalty = 0.0
        if new_portfolio_value < old_portfolio_value:
            drawdown_penalty = ((old_portfolio_value - new_portfolio_value) / (old_portfolio_value + 1e-7)) * 0.2
        
        
        transaction_penalty = 0.0
        if action in [0, 1]:
            transaction_penalty = self.transaction_cost * 10.0
        
        total_reward = base_reward - drawdown_penalty - transaction_penalty
        total_reward = np.clip(total_reward, -100, 100)
        
        obs = self._get_observation()
        info = {
            'balance': self.balance,
            'owned_shares': self.owned_shares,
            'current_price': current_price,
            'portfolio_value': new_portfolio_value
        }
        return obs, total_reward, done, info
    
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.owned_shares = 0.0
        return self._get_observation()
    
    def render(self, mode='human'):
        current_price = self.df.loc[self.current_step, 'Close_Price']
        portfolio_value = self.balance + self.owned_shares * current_price
        profit = portfolio_value - self.initial_balance
        print(f"Step: {self.current_step}")
        print(f"Balance: {self.balance:.2f}, Owned Shares: {self.owned_shares:.2f}")
        print(f"Current Price: {current_price:.2f}")
        print(f"Portfolio Value: {portfolio_value:.2f}, Profit: {profit:.2f}\n")
