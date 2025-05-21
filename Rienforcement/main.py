import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Data_load import load_and_preprocess_data
from Env import StockTradingEnv
from BDQN import BDQNAgent, train_bdqn

def evaluate_model(env, agent):
    state = env.reset()
    done = False
    portfolio_values = []
    rewards = []
    actual_prices = []
    predicted_prices = []
    actions_taken = []
    
    while not done:
    
        obs = state.cpu().numpy() 
        actual_prices.append(obs[1])
        predicted_prices.append(obs[6])
        
        action = agent.select_action(state)
        actions_taken.append(action)
        next_state, reward, done, info = env.step(action)
        portfolio_values.append(info['portfolio_value'])
        rewards.append(reward)
        state = next_state

    
    cumulative_return = (portfolio_values[-1] / env.initial_balance) - 1
    running_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (running_max - portfolio_values) / running_max
    max_drawdown = np.max(drawdowns)
    rewards_array = np.array(rewards)
    sharpe_ratio = rewards_array.mean() / (rewards_array.std() + 1e-7)
    
    print("Evaluation Metrics:")
    print(f"Cumulative Return: {cumulative_return*100:.2f}%")
    print(f"Maximum Drawdown: {max_drawdown*100:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    
    return portfolio_values, rewards, actual_prices, predicted_prices, actions_taken

def main():
    file_path = '/Users/Garry/Year3/Third-Year-Project/Data/Tesla Stock Price History.csv'
    train_df, test_df, scaler = load_and_preprocess_data(file_path, split_ratio=0.8, window_size=20)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_env = StockTradingEnv(train_df, initial_balance=10000, trade_fraction=0.1, device=device)
    test_env = StockTradingEnv(test_df, initial_balance=10000, trade_fraction=0.1, device=device)
    
    
    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.n  
    feature_dim = 32  
    agent = BDQNAgent(state_dim, action_dim, feature_dim, device,
                      hidden_dim=64, lr=1e-3, gamma=0.99, buffer_capacity=10000, batch_size=64,
                      prior_var=1.0, noise_var=0.1, var_k=1.0, bayes_smoothing=0.9)
    num_episodes = 300
    total_rewards, cumulative_returns = train_bdqn(train_env, agent, num_episodes=num_episodes, 
                                                     bayes_update_freq=5, target_update_freq=10)
    agent.save("/Users/Garry/Year3/Third-Year-Project/model/bdqn_trading_model2.pt")
    print("Model saved.")

    test_agent = BDQNAgent(state_dim, action_dim, feature_dim, device,
                           hidden_dim=64, lr=1e-3, gamma=0.99, buffer_capacity=10000, batch_size=64,
                           prior_var=1.0, noise_var=0.1, var_k=1.0, bayes_smoothing=0.9)
    test_agent.load("/Users/Garry/Year3/Third-Year-Project/model/bdqn_trading_model2.pt")
    print("Model loaded for testing.")

    portfolio_values, test_rewards, actual_prices, predicted_prices, actions_taken = evaluate_model(test_env, test_agent)

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(portfolio_values, label='Portfolio Value')
    plt.title("Portfolio Value Over Test Period")
    plt.xlabel("Time Step")
    plt.ylabel("Portfolio Value")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(actual_prices, label='Actual Normalized Price')
    plt.plot(predicted_prices, label='Predicted Price', linestyle='--')
    plt.title("Actual Price vs. Predicted Price")
    plt.xlabel("Time Step")
    plt.ylabel("Normalized Price")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    total_test_reward = np.sum(test_rewards)
    avg_test_reward = np.mean(test_rewards)
    print(f"Total Test Reward: {total_test_reward:.4f}")
    print(f"Average Test Reward per Step: {avg_test_reward:.4f}")

if __name__ == "__main__":
    main()
