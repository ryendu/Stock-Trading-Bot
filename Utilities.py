import gym
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Input, Activation, Concatenate
from tensorflow.keras.optimizers import Adam
from stable_baselines3 import A2C
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from tensorflow import keras
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import yfinance as yf
import gym_anytrading
import rl
from StockTradeEnv import FastTrainingStockTradeEnv
import quantstats as qs

def eval_model(stk_ticker,model,verbose=0):
    """ evaluates a rl ml trading bot using the FastTrainingStockTradeEnv. """
    env = FastTrainingStockTradeEnv(ticker=stk_ticker,frame_bound=(100, 10000000), window_size=10,initial_balance=25000,verbose=verbose)
    observation = env.reset()
    itr = 0
    while True:
        observation = observation
        action, _states = model.predict(observation)
        observation, reward, done, info = env.step(action)
        itr+=1
        if done:
            if verbose != 0:
                print("info:", info)
            print("iters:",itr)
            break
    return env
            
def eval_env_random_sample(stk_ticker,iter_manual=30,verbose=0):
    """ Allows for an environment to be randomly sampled with random actions and evaluated without bias. """
    env = FastTrainingStockTradeEnv(ticker=stk_ticker,frame_bound=(100, 10000000), window_size=10,initial_balance=25000,verbose=verbose)
    observation = env.reset()
    itr = 0
    while itr <= iter_manual:
        observation = observation
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        itr+=1
        if itr == iter_manual:
            if verbose != 0:
                print("info:", info)
            print("iters:",itr)
            break

def plt_actions(history,plot_loc,along_with_key='shares_owned',include_mean_rewards_in_title=False):
    acth = np.array([history['action_type']]).reshape(-1)
    shares = np.array([history[along_with_key]]).reshape(-1)
    hold_x=[]
    buy_x=[]
    sell_x=[]
    hold_y=[]
    buy_y=[]
    sell_y=[]
    for i in range(len(acth)):
        if acth[i] == "Hold":
            hold_x.append(i)
            hold_y.append(shares[i])
        elif acth[i] == "Buy":
            buy_x.append(i)
            buy_y.append(shares[i])
        elif acth[i] == "Sell":
            sell_x.append(i)
            sell_y.append(shares[i])
    plot_loc.scatter(hold_x,hold_y,c='r',label='Hold')
    plot_loc.scatter(buy_x,buy_y,c='g',label='Buy')
    plot_loc.scatter(sell_x,sell_y,c='b',label='Sell')
    plot_loc.legend()
    if include_mean_rewards_in_title:
        return f" Mean Hold Reward: {np.mean(np.array(hold_y))} \n Mean Sell Reward: {np.mean(np.array(sell_y))} \n Mean Buy Reward: {np.mean(np.array(buy_y))} \n"

    
def plot_history(history,size):
    """ plots the history of a RL ml trading bot for better analysis. """
    fig, axs = plt.subplots(3, 2)  # a figure with a 2x2 grid of Axes
    axs[0,0].plot(history['networth'],'r')
    axs[0,0].set_title("Networth")
    axs[0,1].plot(history['shares_owned'],'y',label="Shares Owned")
    plt_actions(history,axs[0,1])
    axs[0,1].set_title("Shares Owned")
    axs[1,0].plot(history['price'],'g')
    axs[1,0].set_title("Stock Price")
    axs[1,1].plot(history['reward'],'y',label="reward")
    axs[1,1].set_title("Reward")
    txt = plt_actions(history,axs[1,1],along_with_key='reward',include_mean_rewards_in_title=True)
    txt = txt + f" Final Networth: ${history['networth'][-1]} \n Final Shares owned: {history['shares_owned'][-1]} \n"
    axs[2,0].hist(history['action_type'])
    axs[2,0].set_title("Action Porportion Hist")
    axs[2,1].set_title("More Information")
    font = {'family': 'DejaVu Sans',
        'color':  'black',
        'weight': 'normal',
        'size': 17,
        }
    axs[2,1].text(s=txt,x=0.2,y=0.2,fontdict=font)
    fig.set_size_inches(size)
    return fig





