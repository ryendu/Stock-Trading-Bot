import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import yfinance as yf
from gym_anytrading.envs import StocksEnv
import rl
from gym_anytrading.envs.trading_env import TradingEnv
from enum import Enum
from gym import spaces
import gym
from gym.utils import seeding
import math

class Actions(Enum):
    Hold = 0
    Buy = 1
    Sell = 2

class FastTrainingStockTradeEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, ticker, window_size, frame_bound,df=None,initial_balance=25000,verbose=1):
        #verbose 1 = only print every 1000 steps along with some meta data. verbose 0 = print nothing. verbose 2 = print everything
        self.stk_ticker = ticker
        if df == None:
            df = yf.Ticker(ticker).history(period="max")
        #sets balance and shares owned, and net worth
        self.INITIAL_BALANCE = initial_balance
        self.current_networth = self.INITIAL_BALANCE
        self.current_balance = self.INITIAL_BALANCE
        self.verbose = verbose
        self.shares_owned = 0
        #sets frame_bound
        assert len(frame_bound) == 2
        self.frame_bound = frame_bound
        assert df.ndim == 2
        #resets random np seed to be more random and consitent 
        self.seed()
        #sets df, window size
        self.df = df
        self.window_size = window_size
        #gets the prices(array of just closing prices from df) and signal features which is the difference between closing prices
        self.prices, self.signal_features = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1])

        self.action_space = spaces.Box(low=np.array([0,0,0,0]), high=np.array([1,1,1,np.inf]), dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=self.shape,
                                            dtype=np.float32)

        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._done = None
        #which step / tick / time step of the training data we are on
        self._current_tick = None
        # the last step / tick / time step of the training data
        self._last_trade_tick = None
        self._total_reward = None
        self._total_profit = 0
        self._first_rendering = None
        self.history = None
        self.previous_trade_stock_price = self.prices[0]
        self.previous_networth = self.INITIAL_BALANCE

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.current_networth = self.INITIAL_BALANCE
        self.current_balance = self.INITIAL_BALANCE
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._total_reward = 0.
        self._total_profit = 0
        self._first_rendering = True
        self.history = {}
        self.shares_owned = 0
        self.previous_trade_stock_price = self.prices[0]
        self.previous_networth = self.INITIAL_BALANCE
        return self._get_observation()

    def update_networth(self):
        current_price = self.prices[self._current_tick]
        self.current_networth = self.current_balance + (self.shares_owned * current_price)
    #TODO: Remake the step function to my desire!

    def step(self, action):
        if self.verbose == 2:
            print(f"Step: {self._current_tick} ----------------------")
        if self.verbose == 1 and self._current_tick % 1000 == 0:
            print(f"Step: {self._current_tick}, current_networth: {self.current_networth} ----------------------")
        self._done = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._done = True
        action_type = np.argmax(np.array(action[:3]))
        action_value = (action[-1])

        trade = False
        if action_type != Actions.Hold.value:
            trade = True

        if trade:
            self._last_trade_tick = self._current_tick
        add_to_step_reward = self.perform_action(action_type, action_value)
        #calculate reward
        current_price = self.prices[self._current_tick]
        
        self.update_networth()
        step_reward = (self.current_networth - self.previous_networth) + add_to_step_reward
        if self.verbose == 2:
            print(f"       step reward: {step_reward}, og step reward: {self.current_networth - self.previous_networth}, add_to_step_reward: {add_to_step_reward}")
        
        self.previous_networth = self.current_networth
        self._total_reward += step_reward
        observation = self._get_observation()
        action_to_str = ["Hold","Buy","Sell"]
        info = {"networth": self.current_networth,"action_type":action_to_str[action_type],"action_value":round(action_value),"tick":self._current_tick,"shares_owned":self.shares_owned,"price":current_price,'reward':step_reward}
        self._update_history(info)
        return observation, step_reward, self._done, info

    def perform_action(self, action_type, action_value):
        add_to_step_reward = 0
        if action_type == Actions.Buy.value:
            current_price = self.prices[self._current_tick]
            max_shares_purchasable = math.floor( self.current_balance / current_price )
            stocks_to_buy = round(action_value)
            if stocks_to_buy > max_shares_purchasable:
                stocks_to_buy = max_shares_purchasable
            if stocks_to_buy >= 1:
                # add_to_step_reward = 1
                self.current_balance -= stocks_to_buy * current_price
                self.shares_owned += stocks_to_buy
                self.previous_trade_stock_price = current_price
                if self.verbose == 2:
                    print(f"        {self.stk_ticker}: Bought {stocks_to_buy} stock(s) for:{current_price}, networth: {self.current_networth}, shares: {self.shares_owned}")
            else:
                if self.verbose == 2:
                    print(f"        Tried to Buy. Unable.")
                add_to_step_reward = -2
        elif action_type == Actions.Sell.value:
            current_price = self.prices[self._current_tick]
            stocks_to_sell = round(action_value)
            if stocks_to_sell > self.shares_owned:
                stocks_to_sell = self.shares_owned
            if stocks_to_sell >= 1:
                # add_to_step_reward = 1
                self.current_balance += stocks_to_sell * current_price
                self.shares_owned -= stocks_to_sell
                add_to_step_reward = (current_price - self.previous_trade_stock_price) * 10
                if self.verbose == 2:
                    print(f"        {self.stk_ticker}: Sold {stocks_to_sell} stock(s) for:{current_price}, networth: {self.current_networth}, shares: {self.shares_owned}")
            else:
                add_to_step_reward = -2
                if self.verbose == 2:
                    print(f"        Tried to Sell. Unable.")
        else:
            if self.shares_owned >= 1 :
                current_price = self.prices[self._current_tick]
                add_to_step_reward = (current_price - self.previous_trade_stock_price)
            else:
                add_to_step_reward = -10
                if self.verbose == 2:
                    print("Discouraging holding")
            if self.verbose == 2:
                print(f"        {self.stk_ticker}: Held, networth: {self.current_networth}, shares: {self.shares_owned}")

        return add_to_step_reward


    def _get_observation(self):
        return self.signal_features[(self._current_tick -
                                     self.window_size):self._current_tick]

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def render(self, mode='human'):
        # def _plot_position(position, tick):
        #     color = None
        #     if position == Positions.Short:
        #         color = 'red'
        #     elif position == Positions.Long:
        #         color = 'green'
        #     if color:
        #         plt.scatter(tick, self.prices[tick], color=color)

        # if self._first_rendering:
        #     self._first_rendering = False
        #     plt.cla()
        #     plt.plot(self.prices)
        #     start_position = self._position_history[self._start_tick]
        #     _plot_position(start_position, self._start_tick)

        # _plot_position(self._position, self._current_tick)

        plt.suptitle("Total Reward: %.6f" % self._total_reward + ' ~ ' +
                     "Total Profit: %.6f" % self._total_profit + ' ~ ' + f"Current Balance: ${self.current_balance}")

        plt.pause(0.01)

    def render_all(self, mode='human'):
        # window_ticks = np.arange(len(self._position_history))
        # plt.plot(self.prices)

        # short_ticks = []
        # long_ticks = []
        # for i, tick in enumerate(window_ticks):
        #     if self._position_history[i] == Positions.Short:
        #         short_ticks.append(tick)
        #     elif self._position_history[i] == Positions.Long:
        #         long_ticks.append(tick)

        # plt.plot(short_ticks, self.prices[short_ticks], 'ro')
        # plt.plot(long_ticks, self.prices[long_ticks], 'go')

        plt.suptitle("Total Reward: %.6f" % self._total_reward + ' ~ ' +
                     "Total Profit: %.6f" % self._total_profit + ' ~ ' + f"Current Balance: ${self.current_balance}")

    def close(self):
        plt.close()

    def save_rendering(self, filepath):
        plt.savefig(filepath)

    def pause_rendering(self):
        plt.show()

    def _process_data(self):
        prices = self.df.loc[:, 'Close'].to_numpy()

        prices[self.frame_bound[0] -
               self.window_size]  # validate index (TODO: Improve validation)
        prices = prices[self.frame_bound[0] -
                        self.window_size:self.frame_bound[1]]

        diff = np.insert(np.diff(prices), 0, 0)
        signal_features = np.column_stack((prices, diff))
        return prices, signal_features

    def max_possible_profit(self):
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.

        while current_tick <= self._end_tick:
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (current_tick <= self._end_tick
                       and self.prices[current_tick] <
                       self.prices[current_tick - 1]):
                    current_tick += 1
            else:
                while (current_tick <= self._end_tick
                       and self.prices[current_tick] >=
                       self.prices[current_tick - 1]):
                    current_tick += 1

            current_price = self.prices[current_tick - 1]
            last_trade_price = self.prices[last_trade_tick]
            shares = profit / last_trade_price
            profit = shares * current_price
            last_trade_tick = current_tick - 1

        return profit
