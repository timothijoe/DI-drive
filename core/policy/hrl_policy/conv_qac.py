from torch import nn
from typing import Union, Dict, Optional, List
from easydict import EasyDict
import torch

from ding.utils import SequenceType, squeeze
from ding.model.template import QAC, VAC
from ding.model.common import RegressionHead, ReparameterizationHead, FCEncoder, DiscreteHead, MultiHead
#from core.models.common_model import ConvEncoder
from core.policy.hrl_policy.traj_qac import ConvEncoder

class ConvQAC(QAC):

    def __init__(
        self,
        obs_shape: Union[int, SequenceType],
        action_shape: Union[int, SequenceType, EasyDict],
        action_space: str,
        encoder_hidden_size_list: SequenceType = [64],
        twin_critic: bool = False,
        actor_head_hidden_size: int = 64,
        actor_head_layer_num: int = 1,
        critic_head_hidden_size: int = 64,
        critic_head_layer_num: int = 1,
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
    ):
        super(QAC, self).__init__()
        obs_shape: int = squeeze(obs_shape)
        action_shape = squeeze(action_shape)
        if isinstance(obs_shape, int) or len(obs_shape) == 1:
            encoder_cls = FCEncoder
        elif len(obs_shape) == 3:
            encoder_cls = ConvEncoder
        else:
            raise RuntimeError(
                "not support obs_shape for pre-defined encoder: {}, please customize your own DQN".format(obs_shape)
            )

        self.action_shape = action_shape
        self.action_space = action_space
        assert self.action_space in ['regression', 'reparameterization', 'hybrid']
        if self.action_space == 'regression':  # DDPG, TD3
            self.actor = nn.Sequential(
                encoder_cls(obs_shape, encoder_hidden_size_list, activation=None, norm_type=norm_type), activation,
                RegressionHead(
                    actor_head_hidden_size,
                    action_shape,
                    actor_head_layer_num,
                    final_tanh=True,
                    activation=activation,
                    norm_type=norm_type
                )
            )
        elif self.action_space == 'reparameterization':  # SAC
            self.actor = nn.Sequential(
                encoder_cls(obs_shape, encoder_hidden_size_list, activation=None, norm_type=norm_type), activation,
                ReparameterizationHead(
                    actor_head_hidden_size,
                    action_shape,
                    actor_head_layer_num,
                    sigma_type='conditioned',
                    activation=activation,
                    norm_type=norm_type
                )
            )
        self.twin_critic = twin_critic
        if self.twin_critic:
            self.critic_encoder = nn.ModuleList()
            self.critic_head = nn.ModuleList()
            for _ in range(2):
                self.critic_encoder.append(
                    encoder_cls(obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type)
                )
                self.critic_head.append(
                    RegressionHead(
                        critic_head_hidden_size + action_shape,
                        1,
                        critic_head_layer_num,
                        final_tanh=False,
                        activation=activation,
                        norm_type=norm_type
                    )
                )
        else:
            self.critic_encoder = encoder_cls(
                obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type
            )
            self.critic_head = RegressionHead(
                critic_head_hidden_size + action_shape,
                1,
                critic_head_layer_num,
                final_tanh=False,
                activation=activation,
                norm_type=norm_type
            )
        if self.twin_critic:
            self.critic = nn.ModuleList([*self.critic_encoder, *self.critic_head])
        else:
            self.critic = nn.ModuleList([self.critic_encoder, self.critic_head])

    def compute_critic(self, inputs: Dict) -> Dict:
        if self.twin_critic:
            x = [m(inputs['obs']) for m in self.critic_encoder]
            x = [torch.cat([x1, inputs['action']], dim=1) for x1 in x]
            x = [m(xi)['pred'] for m, xi in [(self.critic_head[0], x[0]), (self.critic_head[1], x[1])]]
        else:
            x = self.critic_encoder(inputs['obs'])
            x = self.critic_head(x)['pred']
        return {'q_value': x}
