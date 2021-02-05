from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

class ReshapeLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        print(size_out)

    def forward(self, x):
        reshaped=x.reshape(self.size_out)
        return reshaped

class GetFromTupleLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[0]

class LSTMNetwork(nn.Module):
    """
    LSTM network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: tuple,
        last_layer_dim_pi: int = 1,
        last_layer_dim_vf: int = 1,
    ):
        super(LSTMNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        hidden_=20
        layrs=2
        # Policy network
        self.policy_net = nn.Sequential(
            ReshapeLayer(feature_dim,(-1,1,feature_dim)),
            nn.LSTM(input_size=feature_dim, hidden_size=hidden_, num_layers=layrs, bidirectional=False),
            GetFromTupleLayer(),
            nn.ReLU(),
            nn.Linear(hidden_, last_layer_dim_pi), 
            nn.ReLU(),
            GetFromTupleLayer()
        )
        # Value network
        self.value_net = nn.Sequential(
            ReshapeLayer(feature_dim,(-1,1,feature_dim)),
            nn.LSTM(input_size=feature_dim, hidden_size=hidden_, num_layers=layrs, bidirectional=False),
            GetFromTupleLayer(),
            nn.ReLU(),
            nn.Linear(hidden_, last_layer_dim_vf), 
            nn.ReLU(),
            GetFromTupleLayer()
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        pol=self.policy_net(features)
        val=self.value_net(features)
        return pol, val

class LSTMActorCriticPolicy(ActorCriticPolicy):
    Recurrent=True
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):

        super(LSTMActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        # self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        feature_dim=2
        self.mlp_extractor = LSTMNetwork(feature_dim)
