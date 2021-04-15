from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from torch import nn
from torch.nn import functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomLSTM(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 2,
        last_layer_dim_pi: int = 2,
        last_layer_dim_vf: int = 2,

    ):
        super(CustomLSTM, self).__init__(observation_space, features_dim)

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        hidden_=128 
        layers=3

        # self.extract_net = nn.Sequential(
        #     ReshapeLayer(features_dim,(-1,1,features_dim)),
        #     nn.LSTM(input_size=features_dim, hidden_size=hidden_, num_layers=layers, bidirectional=False),
        #     GetFromTupleLayer(),
        #     nn.ReLU(),
        #     nn.Linear(hidden_, last_layer_dim_vf), 
        #     nn.ReLU(),
        #     GetFromTupleLayer()
        # )

        self.lstm1 = nn.LSTM(features_dim, hidden_, layers)
        self.linear1 = nn.Linear(hidden_,hidden_)
        self.linear2 = nn.Linear(hidden_,2)
        self.flatten = nn.Flatten() 
        #(num_layers * num_directions, batch, hidden_size), (num_layers * num_directions, batch, hidden_size)
        self.state = (th.randn(layers, 1, hidden_),th.randn(layers, 1, hidden_))
 
    def forward(self, features: th.Tensor) -> th.Tensor:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        out, self.state=self.lstm1(features, self.state)
        out = F.relu(out)
        out = self.linear1(out)
        out = F.relu(out)
        out = self.linear2(out)
        out = out.view(-1,2)
        return out

# class ReshapeLayer(nn.Module):
#     """ Custom Linear layer but mimics a standard linear layer """
#     def __init__(self, size_in, size_out):
#         super().__init__()
#         self.size_in, self.size_out = size_in, size_out

#     def forward(self, x):
#         reshaped=x.reshape(self.size_out)
#         return reshaped

# class GetFromTupleLayer(nn.Module):
#     """ Custom Linear layer but mimics a standard linear layer """
#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         return x[0]

# class LSTMNetwork(nn.Module):
#     """
#     LSTM network for policy and value function.
#     It receives as input the features extracted by the feature extractor.

#     :param features_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
#     :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
#     :param last_layer_dim_vf: (int) number of units for the last layer of the value network
#     """

#     def __init__(
#         self,
#         features_dim: tuple,
#         last_layer_dim_pi: int = 64,
#         last_layer_dim_vf: int = 64,
#     ):
#         super(LSTMNetwork, self).__init__()

#         # IMPORTANT:
#         # Save output dimensions, used to create the distributions
#         self.latent_dim_pi = last_layer_dim_pi
#         self.latent_dim_vf = last_layer_dim_vf
#         hidden_=20
#         layrs=2
#         # Policy network
#         self.policy_net = nn.Sequential(
#             ReshapeLayer(features_dim,(-1,1,features_dim)),
#             nn.LSTM(input_size=features_dim, hidden_size=hidden_, num_layers=layrs, bidirectional=False),
#             GetFromTupleLayer(),
#             nn.ReLU(),
#             nn.Linear(hidden_, last_layer_dim_pi), 
#             nn.ReLU(),
#             GetFromTupleLayer()
#         )
#         # Value network
#         self.value_net = nn.Sequential(
#             ReshapeLayer(features_dim,(-1,1,features_dim)),
#             nn.LSTM(input_size=features_dim, hidden_size=hidden_, num_layers=layrs, bidirectional=False),
#             GetFromTupleLayer(),
#             nn.ReLU(),
#             nn.Linear(hidden_, last_layer_dim_vf), 
#             nn.ReLU(),
#             GetFromTupleLayer()
#         )

#         # self.policy_net = nn.Sequential(
#         #     nn.Linear(features_dim, last_layer_dim_pi), nn.ReLU()
#         # )
#         # # Value network
#         # self.value_net = nn.Sequential(
#         #     nn.Linear(features_dim, last_layer_dim_vf), nn.ReLU()
#         # )

#     def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
#         """
#         :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
#             If all layers are shared, then ``latent_policy == latent_value``
#         """
#         pol=self.policy_net(features)
#         val=self.value_net(features)
#         return pol, val

# class LSTMActorCriticPolicy(ActorCriticPolicy):
#     Recurrent=True
#     def __init__(
#         self,
#         observation_space: gym.spaces.Space,
#         action_space: gym.spaces.Space,
#         lr_schedule: Callable[[float], float],
#         net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
#         activation_fn: Type[nn.Module] = nn.Tanh,
#         *args,
#         **kwargs,
#     ):

#         super(LSTMActorCriticPolicy, self).__init__(
#             observation_space,
#             action_space,
#             lr_schedule,
#             net_arch,
#             activation_fn,
#             # Pass remaining arguments to base class
#             *args,
#             **kwargs,
#         )
#         # Disable orthogonal initialization
#         # self.ortho_init = False

#     def _build_mlp_extractor(self) -> None:
#         features_dim=2
#         self.mlp_extractor = LSTMNetwork(features_dim)
