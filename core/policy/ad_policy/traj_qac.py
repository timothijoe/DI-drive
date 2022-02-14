from typing import Union, Dict, Optional
from easydict import EasyDict
import numpy as np
import torch
import torch.nn as nn

from ding.utils import SequenceType, squeeze, MODEL_REGISTRY
from ding.model.common import RegressionHead, ReparameterizationHead, DiscreteHead, MultiHead, \
    FCEncoder, ConvEncoder
from core.policy.ad_policy.traj_vae import VaeDecoder
from core.policy.ad_policy.traj_vae import WpDecoder

import torch.nn as nn
from torch.nn import init
#define the initial function to init the layer's parameters for the network
def weigth_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data,0.1)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0,0.01)
        m.bias.data.zero_()




class BEVSpeedConvEncoder(nn.Module):
    def __init__(
        self,
        obs_shape = [200, 200, 5],
        encoder_hidden_size_list = [128, 128, 64],
        embedding_size = 64,
    ):
        super().__init__()
        assert len(obs_shape)==3
        self._obs_shape = obs_shape 
        self._embedding_size = embedding_size
        self._relu = nn.ReLU()
        self.conv_encoder = ConvEncoder(obs_shape, encoder_hidden_size_list)
        self.conv_encoder = nn.Sequential(self.conv_encoder, nn.Flatten())
        flatten_size = self._get_flatten_size()
        self.speed_spd_size = self._embedding_size - self._embedding_size // 2 
        #self.linear_encoder = nn.Sequential(nn.Linear(self.speed_spd_size, self.speed_spd_size), self._relu)
        self._mid = nn.Linear(flatten_size, self._embedding_size // 2)

    def _get_flatten_size(self) -> int:
        test_data = torch.randn(1, *self._obs_shape)
        with torch.no_grad():
            output = self.conv_encoder(test_data)
        return output.shape[1]  

    def forward(self, data: Dict) -> torch.Tensor:
        image = data['birdview']
        speed = data['speed']
        x = self.conv_encoder(image)
        x = self._mid(x)
        speed_vec = torch.unsqueeze(speed, 1).repeat(1, self.speed_spd_size)  
        h = torch.cat((x, speed_vec), dim=1)  
        h = h.to(torch.float32)
        return h 

@MODEL_REGISTRY.register('traj_qac')
class TrajQAC(nn.Module):
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
            share_encoder: bool = True,
            encoder_hidden_size_list: SequenceType = [128, 128, 64],
            embedding_size = 64,
            freeze_decoder = True,
            vae_embedding_dim = 64,
            vae_h_dim = 64,
            vae_latent_dim = 100,
            vae_seq_len = 30,
            vae_dt = 0.03,
            vae_load_dir = None,
            use_wp_decoder = False,
            twin_critic: bool = False,
            actor_head_hidden_size: int = 64,
            actor_head_layer_num: int = 3,
            critic_head_hidden_size: int = 64,
            critic_head_layer_num: int = 3,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None,
            
            
    ) -> None:
        """
        Overview:
            Init the QAC Model according to arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation's space.
            - action_shape (:obj:`Union[int, SequenceType, EasyDict]`): Action's space, such as 4, (3, ), \
                EasyDict({'action_type_shape': 3, 'action_args_shape': 4}).
                zt -comment: here we denote the action shape as the latent action
                and if we output trajectory, the critic input shape should be different
            - action_space (:obj:`str`): Whether choose ``regression`` or ``reparameterization`` or ``hybrid`` .

        """
        super(TrajQAC, self).__init__()
        obs_shape: int = squeeze(obs_shape)
        action_shape = squeeze(action_shape)
        self.action_shape = action_shape
        self.action_space = action_space
        self.freeze_decoder = freeze_decoder
        self.use_wp_decoder = use_wp_decoder
        self.embedding_size = embedding_size
        # Here we will not train decoder but load the perfect one
        assert freeze_decoder == True
        # Here we regard the output latent is the action, so we assume they are the sasme
        assert vae_latent_dim == action_shape

        # Observation Encoder part:
        if isinstance(obs_shape, int) or len(obs_shape) == 1:
            encoder_cls = FCEncoder
        elif len(obs_shape) == 3:
            encoder_cls = BEVSpeedConvEncoder
        else:
            raise RuntimeError(
                "not support obs_shape for pre-defined encoder: {}, please customize your own DQN".format(obs_shape)
            )
        self.share_encoder = share_encoder
        if self.share_encoder:
            self.encoder = encoder_cls(obs_shape, encoder_hidden_size_list)
        else:
            self.actor_encoder = encoder_cls(
                obs_shape, encoder_hidden_size_list
            )
            self.critic_encoder = encoder_cls(
                obs_shape, encoder_hidden_size_list
            )
        
        # VAE part, 
        self.traj_len = vae_seq_len
        if freeze_decoder:
            # Here action shape means latent space shape
            critic_head_hidden_size = self.action_shape + embedding_size
        else:
            # traj_len + 1 means adding the inital state getting from observation
            # # 2 means x, y for each point
            critic_head_hidden_size = 2 * (self.traj_len + 1) + embedding_size
        
        # critic part
        self.critic_head = RegressionHead(
            critic_head_hidden_size, 1, critic_head_layer_num, activation=activation, norm_type=norm_type
        )

        assert self.action_space in ['discrete', 'regression', 'reparameterization']
        assert self.action_space == 'reparameterization'

        if self.action_space == 'reparameterization':
            self.actor_head = ReparameterizationHead(
                actor_head_hidden_size,
                action_shape,
                actor_head_layer_num,
                sigma_type='conditioned',
                activation=activation,
                norm_type=norm_type              
            )
        self.actor = nn.Sequential(self.encoder, self.actor_head)
        self.critic = nn.ModuleList([self.encoder, self.critic_head])
        self.actor.apply(weigth_init)
        self.critic.apply(weigth_init)
        # If so, use wp decoder, we only output waypoint, which is not a lstm decoder but dynamic bycicle model
        if self.use_wp_decoder:
            self._traj_decoder = WpDecoder(
                seq_len = vae_seq_len,
                dt = vae_dt
            )
        else:
            self._traj_decoder = VaeDecoder(
                embedding_dim = vae_embedding_dim,
                h_dim = vae_h_dim,
                latent_dim = vae_latent_dim,
                seq_len = vae_seq_len,
                dt = vae_dt
            )
        if vae_load_dir is not None:
            self._traj_decoder.load_state_dict(torch.load(vae_load_dir))


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
        if self.action_space == 'reparameterization':
            x = self.actor(inputs)
            return {'logit': [x['mu'], x['sigma']]}
        
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

        obs, latent_action, traj = inputs['obs'], inputs['latent_action'], inputs['trajectory']
        lat_obs = self.encoder(obs)
        if self.freeze_decoder:
            # if we freeze decoder, we use latent action to judge the q value
            critic_input = torch.cat([lat_obs, latent_action], dim = 1)
        else:
            traj_input = traj.contiguous().view(-1, 2 * (self.traj_len + 1))
            critic_input = torch.cat([lat_obs, traj_input], dim = 1)
        x = self.critic_head(critic_input)['pred']
        return {'q_value': x}

    def generate_traj_from_lat(self, latent_action, init_state):
        #print('init state: {}'.format(init_state[0]))
        #print('control: {}'.format(latent_action[0]))
        
        traj = self._traj_decoder(latent_action, init_state)
        #print('final_state: {}'.format(traj[0]))
        traj = torch.cat([init_state.unsqueeze(1), traj], dim = 1)
        # for parameters in self._traj_decoder.parameters():
        #     print(parameters)
        return traj[:, :,:2]


         
