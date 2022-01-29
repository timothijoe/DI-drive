from typing import Union, Dict, Optional
from easydict import EasyDict
import numpy as np
import torch
import torch.nn as nn

from ding.utils import SequenceType, squeeze, MODEL_REGISTRY
# from ..common import RegressionHead, ReparameterizationHead, DiscreteHead, MultiHead, \
#     FCEncoder, ConvEncoder
from ding.model.common import ReparameterizationHead, FCEncoder, ConvEncoder
@MODEL_REGISTRY.register('conv_qac')
class ConvQAC(nn.Module):
    r"""
    Overview:
        The QAC model.
    Interfaces:
        ``__init__``, ``forward``, ``compute_actor``, ``compute_critic``
    """
    mode = ['compute_actor', 'compute_critic']

    def __init__(
            self,
            obs_shape: Union[int, SequenceType],
            action_shape: Union[int, SequenceType, EasyDict],
            action_space: str,
            traj_seq_len: int = 30,
            train_decoder: bool = False,
            twin_critic: bool = False,
            actor_head_hidden_size: int = 64,
            actor_head_layer_num: int = 1,
            critic_head_hidden_size: int = 64,
            critic_head_layer_num: int = 1,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None,
            encoder_hidden_size_list: SequenceType = [128, 128, 64],
    ) -> None:
        """
        Overview:
            Init the QAC Model according to arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation's space.
            - action_shape (:obj:`Union[int, SequenceType, EasyDict]`): Action's space, such as 4, (3, ), \
                EasyDict({'action_type_shape': 3, 'action_args_shape': 4}).
            - action_space (:obj:`str`): Whether choose ``regression`` or ``reparameterization`` or ``hybrid`` .
            - twin_critic (:obj:`bool`): Whether include twin critic.
            - actor_head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` to pass to actor-nn's ``Head``.
            - actor_head_layer_num (:obj:`int`): The num of layers used in the network to compute Q value output \
                for actor's nn.
            - critic_head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` to pass to critic-nn's ``Head``.
            - critic_head_layer_num (:obj:`int`): The num of layers used in the network to compute Q value output \
                for critic's nn.
            - activation (:obj:`Optional[nn.Module]`): The type of activation function to use in ``MLP`` \
                after ``layer_fn``, if ``None`` then default set to ``nn.ReLU()``
            - norm_type (:obj:`Optional[str]`): The type of normalization to use, \
                see ``ding.torch_utils.netwrok`` for more details.
        """
        super(ConvQAC, self).__init__()
        obs_shape: int = squeeze(obs_shape)
        action_shape = squeeze(action_shape)
        self.action_shape = action_shape
        self.action_space = action_space
        self.traj_seq_len = traj_seq_len
        if isinstance(obs_shape, int) or len(obs_shape) == 1:
            encoder_cls = FCEncoder
        elif len(obs_shape) == 3:
            encoder_cls = ConvEncoder
        else:
            raise RuntimeError(
                "not support obs_shape for pre-defined encoder: {}, please customize your own DQN".format(obs_shape)
            )
        self.actor_encoder = encoder_cls(obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type)
        self.critic_encoder =  encoder_cls(obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type)
        self.actor_head = ReparameterizationHead(
            encoder_hidden_size_list[-1],
            action_shape,
            actor_head_layer_num,
            sigma_type='conditioned',
            activation=activation,
            norm_type=norm_type
        )
        self.actor = nn.Sequential(self.actor_encoder, self.actor_head)
        self.critic_input_size = encoder_hidden_size_list[-1] + self.action_shape 
        self.critic = ReparameterizationHead(
            self.critic_input_size,
            1,
            critic_head_layer_num,
            sigma_type='conditioned',
            activation = activation,
            norm_type = norm_type
        )

    def forward(self, inputs: Union[torch.Tensor, Dict], mode: str) -> Dict:
        """
        Overview:
            Use observation and action tensor to predict output.
            Parameter updates with QAC's MLPs forward setup.
        Arguments:
            Forward with ``compute_actor``:
                - inputs (:obj:`torch.Tensor`): The encoded embedding tensor, determined with given ``hidden_size``, \
                    i.e. ``(B, N=hidden_size)``.

            Forward with ``compute_critic``:
                - inputs (:obj:`Dict`)

            - mode (:obj:`str`): Name of the forward mode.
        Returns:
            - outputs (:obj:`Dict`): Outputs of network forward.

                Forward with ``compute_actor``
                    - action (:obj:`torch.Tensor`): Action tensor with same size as input ``x``.
                    - logit (:obj:`torch.Tensor`): Logit tensor encoding ``mu`` and ``sigma``, both with same size \
                        as input ``x``.

                Forward with ``compute_critic``
                    - q_value (:obj:`torch.Tensor`): Q value tensor with same size as batch size.
        Actor Shapes:
            - inputs (:obj:`torch.Tensor`): :math:`(B, N0)`, B is batch size and N0 corresponds to ``hidden_size``
            - action (:obj:`torch.Tensor`): :math:`(B, N0)`
            - q_value (:obj:`torch.FloatTensor`): :math:`(B, )`, where B is batch size.

        Critic Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, N1)`, where B is batch size and N1 is ``obs_shape``
            - action (:obj:`torch.Tensor`): :math:`(B, N2)`, where B is batch size and N2 is ``action_shape``
            - logit (:obj:`torch.FloatTensor`): :math:`(B, N2)`, where B is batch size and N3 is ``action_shape``

        Actor Examples:
            >>> # Regression mode
            >>> model = QAC(64, 64, 'regression')
            >>> inputs = torch.randn(4, 64)
            >>> actor_outputs = model(inputs,'compute_actor')
            >>> assert actor_outputs['action'].shape == torch.Size([4, 64])
            >>> # Reparameterization Mode
            >>> model = QAC(64, 64, 'reparameterization')
            >>> inputs = torch.randn(4, 64)
            >>> actor_outputs = model(inputs,'compute_actor')
            >>> actor_outputs['logit'][0].shape # mu
            >>> torch.Size([4, 64])
            >>> actor_outputs['logit'][1].shape # sigma
            >>> torch.Size([4, 64])

        Critic Examples:
            >>> inputs = {'obs': torch.randn(4,N), 'action': torch.randn(4,1)}
            >>> model = QAC(obs_shape=(N, ),action_shape=1,action_space='regression')
            >>> model(inputs, mode='compute_critic')['q_value'] # q value
            tensor([0.0773, 0.1639, 0.0917, 0.0370], grad_fn=<SqueezeBackward1>)

        """
        assert mode in self.mode, "not support forward mode: {}/{}".format(mode, self.mode)
        return getattr(self, mode)(inputs)

    def compute_actor(self, inputs: torch.Tensor) -> Dict:
        """
        Overview:
            Use encoded embedding tensor to predict output.
            Execute parameter updates with ``compute_actor`` mode
            Use encoded embedding tensor to predict output.
        Arguments:
            - inputs (:obj:`torch.Tensor`):
                The encoded embedding tensor, determined with given ``hidden_size``, i.e. ``(B, N=hidden_size)``. \
                ``hidden_size = actor_head_hidden_size``
            - mode (:obj:`str`): Name of the forward mode.
        Returns:
            - outputs (:obj:`Dict`): Outputs of forward pass encoder and head.

        ReturnsKeys (either):
            - action (:obj:`torch.Tensor`): Continuous action tensor with same size as ``action_shape``.
            - logit (:obj:`torch.Tensor`): Logit tensor encoding ``mu`` and ``sigma``, both with same size \
                as input ``x``.
        Shapes:
            - inputs (:obj:`torch.Tensor`): :math:`(B, N0)`, B is batch size and N0 corresponds to ``hidden_size``
            - action (:obj:`torch.Tensor`): :math:`(B, N0)`
            - logit (:obj:`Union[list, torch.Tensor]`):

              - case1(continuous space, list): 2 elements, mu and sigma, each is the shape of :math:`(B, N0)`.
              - case2(hybrid space, torch.Tensor): :math:`(B, N1)`, where N1 is action_type_shape
            - q_value (:obj:`torch.FloatTensor`): :math:`(B, )`, B is batch size.
            - action_args (:obj:`torch.FloatTensor`): :math:`(B, N2)`, where N2 is action_args_shape \
                (action_args are continuous real value)
        Examples:
            >>> # Regression mode
            >>> model = QAC(64, 64, 'regression')
            >>> inputs = torch.randn(4, 64)
            >>> actor_outputs = model(inputs,'compute_actor')
            >>> assert actor_outputs['action'].shape == torch.Size([4, 64])
            >>> # Reparameterization Mode
            >>> model = QAC(64, 64, 'reparameterization')
            >>> inputs = torch.randn(4, 64)
            >>> actor_outputs = model(inputs,'compute_actor')
            >>> actor_outputs['logit'][0].shape # mu
            >>> torch.Size([4, 64])
            >>> actor_outputs['logit'][1].shape # sigma
            >>> torch.Size([4, 64])
        """
        if self.action_space == 'regression':
            x = self.actor_encoder(inputs)
            x = self.actor_head(x)
            #x = self.actor(inputs)
            return {'logit': [x['mu'], x['sigma']]}
        elif self.action_space == 'reparameterization':
            x = self.actor(inputs)
            return {'logit': [x['mu'], x['sigma']]}
        elif self.action_space == 'hybrid':
            logit = self.actor[0](inputs)
            action_args = self.actor[1](inputs)
            return {'logit': logit['logit'], 'action_args': action_args['pred']}

    def compute_critic(self, inputs: Dict) -> Dict:
        r"""
        Overview:
            Execute parameter updates with ``compute_critic`` mode
            Use encoded embedding tensor to predict output.
        Arguments:
            - inputs (:obj:`Dict`): ``obs``, ``action`` and ``logit`` tensors.
            - mode (:obj:`str`): Name of the forward mode.
        Returns:
            - outputs (:obj:`Dict`): Q-value output.

        ArgumentsKeys:
            - necessary:

              - obs: (:obj:`torch.Tensor`): 2-dim vector observation
              - action (:obj:`Union[torch.Tensor, Dict]`): action from actor
            - optional:

              - logit (:obj:`torch.Tensor`): discrete action logit
        ReturnKeys:
            - q_value (:obj:`torch.Tensor`): Q value tensor with same size as batch size.
        Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, N1)`, where B is batch size and N1 is ``obs_shape``
            - action (:obj:`torch.Tensor`): :math:`(B, N2)`, where B is batch size and N2 is ``action_shape``
            - q_value (:obj:`torch.FloatTensor`): :math:`(B, )`, where B is batch size.

        Examples:
            >>> inputs = {'obs': torch.randn(4, N), 'action': torch.randn(4, 1)}
            >>> model = QAC(obs_shape=(N, ),action_shape=1,action_space='regression')
            >>> model(inputs, mode='compute_critic')['q_value']  # q value
            >>> tensor([0.0773, 0.1639, 0.0917, 0.0370], grad_fn=<SqueezeBackward1>)
        """

        obs, action = inputs['obs'], inputs['latent_action']
        lat_obs = self.critic_encoder(obs)
        critic_input = torch.cat([lat_obs, action], dim=1)
        x = self.critic(critic_input)['mu']
        return {'q_value': x}
    
    def compute_traj_critic(self, inputs):
        return 0