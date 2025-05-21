# Tesla Stock Prediction with Bayesian Deep Learning

This project aims to predict Tesla stock prices using two different approaches:
- **Supervised Learning** using a **Bayesian LSTM** model.
- **Reinforcement Learning** using a **Bayesian DQN** model.

Both approaches incorporate Bayesian methods to quantify uncertainty in predictions—allowing for risk-aware trading decisions on volatile data. In each technique, an initial capital of \$10,000 is provided and the model makes buy, sell, or hold decisions. The results are evaluated based on profit/loss and are visualized via figures stored in the `figures/` folder, while all trained models are saved in the `model/` folder.

---

## Project Structure

---

## Models

### Bayesian LSTM (Supervised Learning)
- **Overview:**  
  A Bayesian LSTM model uses Long Short-Term Memory networks enhanced with Bayesian techniques (e.g., Monte Carlo dropout) to predict future stock prices.  
- **How It Works:**  
  The model is trained on historical Tesla stock data with engineered features (e.g., log price, moving averages, returns) and produces both point predictions and uncertainty estimates.  
- **Trading Simulation:**  
  With \$10,000 as starting capital, the model takes active buy/sell decisions based on its prediction—aiming to maximize profit. Results and performance metrics are saved in the `figures/` folder.

### Bayesian DQN (Reinforcement Learning)
- **Overview:**  
  A Bayesian Deep Q-Network (DQN) model learns an optimal trading policy by estimating action-value functions while incorporating uncertainty in the network weights.  
- **How It Works:**  
  Using reinforcement learning libraries (e.g., Gym), the agent interacts with a simulated trading environment for Tesla stock, receiving rewards for profitable trades.  
- **Trading Simulation:**  
  Similarly, the model starts with \$10,000 and decides between buy, sell, or hold actions. The simulation performance is recorded and plotted, with trained models saved in the `model/` folder.

---

## Bayesian vs. Normal Algorithms

- **Bayesian Algorithms:**  
  They model uncertainty by placing probability distributions over weights rather than fixed values. This results in confidence intervals for predictions, which is crucial in volatile markets.
  
- **Normal (Deterministic) Algorithms:**  
  They use fixed weights and provide point estimates only, lacking inherent measures of uncertainty and potentially leading to overconfident decisions.

---
## Data

- **Source:**
The historical data has been collected from YahooFinance website from year 2010 to 2025.  

---
## Requirements

The project uses the following libraries:
- Python 3.x
- **PyTorch**
- **TensorFlow** (and Keras)
- **NumPy**
- **Pandas**
- **Matplotlib**
- **scikit-learn**
- **Gym**

### Credit: ID-11028972  Supervisor: Hongpeng Zhou

Install the required packages via pip:
```bash
pip install torch tensorflow numpy pandas matplotlib scikit-learn gym
 
