from typing import Dict, List, Optional, Any
from collections import namedtuple
import numpy as np
import torch
import torch.nn.functional as F

from ding.utils.data import default_collate, default_decollate
from ding.torch_utils import to_device
from core.models import CILRSModel
from demo.metadrive.base_il_policy import BaseILPolicy
from core.policy.hrl_policy.traj_qac import ConvQAC 
from core.policy.hrl_policy.traj_sac import TrajSAC

class SPiRLPolicy(BaseILPolicy):
    """
    CILRS driving policy. It has a CILRS NN model which can handle
    observations from several environments by collating data into batch. It contains 2
    modes: `eval` and `learn`. The learn mode will calculate all losses, but will not
    back-propregate it. In `eval` mode, the output control signal will be postprocessed to
    standard control signal in Carla, and it can avoid stopping in the staring ticks.

    :Arguments:
        - cfg (Dict): Config Dict.
        - enable_field(List): Enable policy filed, default to ['eval', 'learn']

    :Interfaces:
        reset, forward
    """

    config = dict(
        cuda=True,
        max_throttle=0.75,
        model=dict(),
        learn=dict(
            epoches=200,
            lr=1e-4,
            batch_size=128,
            loss='l1',
            speed_weight=0.05,
            control_weights=[0.5, 0.45, 0.05],
        ),
    )

    def __init__(self, cfg: Dict, model, enable_field: List = ['eval', 'learn']) -> None:
        super().__init__(cfg, enable_field=enable_field)
        self._cuda = self._cfg.cuda
        self._model =model
        if self._cuda:
            self._model.cuda()

        for field in self._enable_field:
            getattr(self, '_init_' + field)()

    # def _process_sensors(self, sensor: np.ndarray) -> np.ndarray:
    #     sensor = sensor[:, :, ::-1]  # BGR->RGB
    #     sensor = np.transpose(sensor, (2, 0, 1))
    #     sensor = sensor / 255.0

    #     return sensor

    # def _process_model_outputs(self, data: Dict, output: List) -> List:
    #     action = []
    #     for i, d in enumerate(data.values()):
    #         control_pred = output[i][0]
    #         steer = control_pred[0] * 2 - 1.  # convert from [0,1] to [-1,1]
    #         throttle = min(control_pred[1], self._max_throttle)
    #         brake = control_pred[2]
    #         if d['tick'] < 20 and d['speed'] < 0.1:
    #             throttle = self._max_throttle
    #             brake = 0
    #         if brake < 0.05:
    #             brake = 0
    #         action.append({'steer': steer, 'throttle': throttle, 'brake': brake})
    #     return action

    def _reset_eval(self, data_id: Optional[List[int]] = None) -> None:
        """
        Reset policy of `eval` mode. It will change the NN model into 'eval' mode.

        :Arguments:
            - data_id (List[int], optional): List of env id to reset. Defaults to None.
        """
        self._model.eval()

    @torch.no_grad()
    def _forward_eval(self, data: Dict) -> Dict[str, Any]:
        """
        Running forward to get control signal of `eval` mode.

        :Arguments:
            - data (Dict): Input dict, with env id in keys and related observations in values,

        :Returns:
            Dict: Control and waypoints dict stored in values for each provided env id.
        """
        data_id = list(data.keys())

        new_data = dict()
        for id in data.keys():
            new_data[id] = dict()
            new_data[id]['rgb'] = self._process_sensors(data[id]['rgb'].numpy())
            new_data[id]['command'] = data[id]['command']
            new_data[id]['speed'] = data[id]['speed']

        new_data = default_collate(list(new_data.values()))
        if self._cuda:
            new_data = to_device(new_data, 'cuda')

        embedding = self._model.encode([new_data['rgb']])
        output = self._model(embedding, new_data['speed'], new_data['command'])
        if self._cuda:
            output = to_device(output, 'cpu')

        actions = default_decollate(output)
        actions = self._process_model_outputs(data, actions)
        return {i: {'action': d} for i, d in zip(data_id, actions)}

    def _init_learn(self) -> None:
        # if self._cfg.learn.loss == 'l1':
        #     self._criterion = F.l1_loss
        # elif self._cfg.policy.learn.loss == 'l2':
        #     self._criterion = F.mse_loss
        self._criterion = F.mse_loss

    def _reset_learn(self, data_id: Optional[List[int]] = None) -> None:
        """
        Reset policy of `learn` mode. It will change the NN model into 'train' mode.

        :Arguments:
            - data_id (List[int], optional): List of env id to reset. Defaults to None.
        """
        self._model.train()

    def _forward_learn(self, data: Dict) -> Dict[str, Any]:
        """
        Running forward of `learn` mode to get loss.

        :Arguments:
            - data (Dict): Input dict, with env id in keys and related observations in values,

        :Returns:
            Dict: information about training loss.
        """


        # data_id = list(data.keys())
        # data = default_collate(list(data.values()))

        if self._cuda:
            data = to_device(data, self._device)

        with torch.no_grad():
            (mu, sigma) = self._model.forward(data, mode='compute_actor')['logit']
            action = torch.tanh(mu)  # deterministic_eval
            output = {'action': action}
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)


        return_info = {
            'total_loss': 0.0,
            'speed_loss': 0.0,
            'steer_loss': 0.0,
            'throttle_loss': 0.0,
            'brake_loss': 0.0,
            # 'steer_mean': steer_pred.item().mean(),
            # 'throttle_mean': throttle_pred.item().mean(),
            # 'brake_mean': brake_pred.item().mean(),
        }

        return return_info
