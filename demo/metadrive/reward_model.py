from collections.abc import Iterable
from easydict import EasyDict
import numpy as np
import pickle
from copy import deepcopy
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Independent
from torch.distributions.categorical import Categorical

from ding.utils import REWARD_MODEL_REGISTRY
from ding.model.template.q_learning import DQN
from ding.model.template.vac import VAC
from ding.model.template.qac import QAC
from ding.utils import SequenceType
from ding.model.common import FCEncoder
from ding.utils.data import offline_data_save_type
from ding.utils import build_logger
from dizoo.atari.envs.atari_wrappers import wrap_deepmind
from dizoo.mujoco.envs.mujoco_wrappers import wrap_mujoco

from ding.reward_model.base_reward_model import BaseRewardModel 
from ding.reward_model.rnd_reward_model import collect_states 
import os 

num_expert_data_to_collect = 10
trex_path = 'test_trex_ad'
trex_reward_folder = trex_path + '/reward_model'
trex_expert_data_folder = trex_path + '/expert_data_folder'
# trex_expert_ckpt_path = trex_path + '/expert_macro_policy/april10_iteration40k.pth.tar'

assert os.path.exists(trex_path)
assert os.path.exists(trex_expert_data_folder)

# if not os.path.exists(trex_expert_data_folder):
#     os.makedirs(trex_expert_data_folder)
if not os.path.exists(trex_reward_folder):
    os.makedirs(trex_reward_folder)





def collect_states(iterator):
    res = []
    for item in iterator:
        state = item['obs']  #item['obs']
        res.append(state)
    return res





class ConvEncoder(nn.Module):
    r"""
    Overview:
        The ``Convolution Encoder`` used in models. Used to encoder raw 2-dim observation.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(
            self,
            obs_shape: SequenceType,
            hidden_size_list: SequenceType = [16, 16, 16, 16, 64, 1],
            activation: Optional[nn.Module] = nn.LeakyReLU(),
            norm_type: Optional[str] = None
    ) -> None:
        r"""
        Overview:
            Init the Convolution Encoder according to arguments.
        Arguments:
            - obs_shape (:obj:`SequenceType`): Sequence of ``in_channel``, some ``output size``
            - hidden_size_list (:obj:`SequenceType`): The collection of ``hidden_size``
            - activation (:obj:`nn.Module`):
                The type of activation to use in the conv ``layers``,
                if ``None`` then default set to ``nn.ReLU()``
            - norm_type (:obj:`str`):
                The type of normalization to use, see ``ding.torch_utils.ResBlock`` for more details
        """
        super(ConvEncoder, self).__init__()
        self.obs_shape = obs_shape
        self.act = activation
        self.hidden_size_list = hidden_size_list

        layers = []
        kernel_size = [7, 5, 3, 3]
        stride = [3, 2, 1, 1]
        input_size = obs_shape[0]  # in_channel
        for i in range(len(kernel_size)):
            layers.append(nn.Conv2d(input_size, hidden_size_list[i], kernel_size[i], stride[i]))
            layers.append(self.act)
            input_size = hidden_size_list[i]
        layers.append(nn.Flatten())
        self.main = nn.Sequential(*layers)

        flatten_size = self._get_flatten_size()
        self.mid = nn.Sequential(
            nn.Linear(flatten_size, hidden_size_list[-2]), self.act,
            nn.Linear(hidden_size_list[-2], hidden_size_list[-1])
        )

    def _get_flatten_size(self) -> int:
        r"""
        Overview:
            Get the encoding size after ``self.main`` to get the number of ``in-features`` to feed to ``nn.Linear``.
        Arguments:
            - x (:obj:`torch.Tensor`): Encoded Tensor after ``self.main``
        Returns:
            - outputs (:obj:`torch.Tensor`): Size int, also number of in-feature
        """
        test_data = torch.randn(1, *self.obs_shape)
        with torch.no_grad():
            output = self.main(test_data)
        return output.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Overview:
            Return embedding tensor of the env observation
        Arguments:
            - x (:obj:`torch.Tensor`): Env raw observation
        Returns:
            - outputs (:obj:`torch.Tensor`): Embedding tensor
        """
        x = self.main(x)
        x = self.mid(x)
        return x

class TrexModel(nn.Module):
    def __init__(self, obs_shape):
        super(TrexModel, self).__init__()
        if isinstance(obs_shape, int) or len(obs_shape) == 1:
            self.encoder = nn.Sequential(FCEncoder(obs_shape, [512, 64]), nn.Linear(64, 1))
        # Conv Encoder
        elif len(obs_shape) == 3:
            self.encoder = ConvEncoder(obs_shape)
        else:
            raise KeyError(
                "not support obs_shape for pre-defined encoder: {}, please customize your own Trex model".
                format(obs_shape)
            )

    
    def cum_return(self, traj: torch.Tensor, mode: str='sum') -> Tuple[torch.Tensor, torch.Tensor]:
        r = self.encoder(traj)
        if mode == 'sum':
            sum_rewards = torch.sum(r)
            sum_abs_rewards = torch.sum(torch.abs(r))
            return sum_rewards, sum_abs_rewards
        elif mode == 'batch': #'epoch'
            return r, torch.abs(r)
        else:
            raise KeyError("not support mode: {}, please choose mode=sum or mode=batch".format(mode))

    def forward(self, traj_i: torch.Tensor, traj_j: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''compute cumulative returns for each trajectory and return logits'''
        cum_r_i, abs_r_i = self.cum_return(traj_i)
        cum_r_j, abs_r_j = self.cum_return(traj_j)
        return torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)), 0), abs_r_i + abs_r_j 

class TrexRewardModelAD(BaseRewardModel):
    config = dict(
        type='trex',
        learning_rate=1e-5,
        update_per_collect=100,
        batch_size=64,
        target_new_data_count=64,
        hidden_size=128,
        num_trajs=0,  # number of downsampled full trajectories
        num_snippets=6000,  # number of short subtrajectories to sample
    )

    #def __init__(self, config: EasyDict, device: str, tb_logger: 'SummaryWriter') -> None:  # noqa
    def __init__(self) -> None:  # noqa
        """
        Overview:
            Initialize ``self.`` See ``help(type(self))`` for accurate signature.
        Arguments:
            - cfg (:obj:`EasyDict`): Training config
            - device (:obj:`str`): Device usage, i.e. "cpu" or "cuda"
            - tb_logger (:obj:`SummaryWriter`): Logger, defaultly set as 'SummaryWriter' for model summary
        """
        super(TrexRewardModelAD, self).__init__()
        #self.cfg = config
        #assert device in ["cpu", "cuda"] or "cuda" in device
        #self.device = device
        #self.tb_logger = tb_logger
        self.reward_model = TrexModel((5,200,200))
        #self.reward_model.to(self.device)
        self.pre_expert_data = []
        self.train_data = []
        self.expert_data_loader = None
        self.opt = optim.Adam(self.reward_model.parameters(), 1e-3)
        self.train_iter = 0
        self.learning_returns = []
        self.learning_rewards = []
        self.training_obs = []
        self.training_labels = []
        # self.num_trajs = self.cfg.reward_model.num_trajs
        # self.num_snippets = self.cfg.reward_model.num_snippets
        # minimum number of short subtrajectories to sample
        self.min_snippet_length = 15
        # maximum number of short subtrajectories to sample
        self.max_snippet_length = 50
        #self.fixed_snippet_length = 15
        self.l1_reg = 0
        self.data_for_save = {}
        self.pickle_folder = trex_expert_data_folder
        self.reward_path = trex_reward_folder + '/reward_ckpt'
        self.expert_data_list = []
        self._logger, self._tb_logger = build_logger(
            path='./{}/log/{}'.format(trex_path, 'trex_reward_model'), name='trex_reward_model'
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.reward_model.to(device)
        #self.load_expert_data()
    
    def _load_expert_folder(self) -> None:
        import os
        file_list = []
        for cur_file in os.listdir(self.pickle_folder):
            cur_file_dir = os.path.join(self.pickle_folder, cur_file)
            file_list.append(cur_file_dir)
        # return file_list 
        for file in file_list:
            with open(file, 'rb') as f:
                episode_data = pickle.load(f)
                self.expert_data_list.append(episode_data)
        print('zt')
        print(len(self.expert_data_list))
        demonstrations = sorted(self.expert_data_list, key = lambda x: x["episode_rwd"], reverse=True)
        damos = []
        for demo in demonstrations:
            damo = []
            for transition in demo['transition_list']:
                damo.append(transition['state'].transpose((2, 0, 1)))
            damos.append(damo)
        self.demonstrations = damos 


    # def _load_expert_data(self) -> None:
    #     with open(self.pickle_path, 'rb') as f:
    #         self.pre_expert_data = pickle.load(f)
    #         print(self.pre_expert_data.keys())
    #         print(self.pre_expert_data['episode_rwd'])
    #         print(len(self.pre_expert_data['transition_list']))
    #         # print(self.pre_expert_data['state_aciton'][0].keys())
    #         # print(self.pre_expert_data['state_action']['obs'].shape)
    #         # print(self.pre_expert_data['state_action']['action'].shape)
    #         # print(self.pre_expert_data['state_action']['reward'].shape)
    
    def create_training_data(self):
        #demonstrations = self.pre_expert_data
        demonstrations = self.demonstrations 
        num_trajs = len(demonstrations)
        num_snippets = 600
        min_snippet_length = self.min_snippet_length
        max_snippet_length = self.max_snippet_length
        demo_lengths = [len(d) for d in demonstrations]
        self._logger.info("demo_lengths: {}".format(demo_lengths))
        max_snippet_length = min(np.min(demo_lengths), max_snippet_length)
        self._logger.info("min snippet length: {}".format(min_snippet_length))
        self._logger.info("max snippet length: {}".format(max_snippet_length))

        #collect training data
        max_traj_length = 0
        num_demos = len(demonstrations)
        assert num_demos >= 2

        # #add full trajs (for use on Enduro)
        # si = np.random.randint(6, size=num_trajs)
        # sj = np.random.randint(6, size=num_trajs)
        # step = np.random.randint(3, 7, size=num_trajs)
        # for n in range(num_trajs):
        #     #pick two random demonstrations
        #     ti, tj = np.random.choice(num_demos, size=(2, ), replace=False)
        #     #create random partial trajs by finding random start frame and random skip frame
        #     traj_i = demonstrations[ti][si[n]::step[n]]  # slice(start,stop,step)
        #     traj_j = demonstrations[tj][sj[n]::step[n]]

        #     label = int(ti <= tj)

        #     self.training_obs.append((traj_i, traj_j))
        #     self.training_labels.append(label)
        #     max_traj_length = max(max_traj_length, len(traj_i), len(traj_j))

        #fixed size snippets with progress prior
        rand_length = np.random.randint(min_snippet_length, max_snippet_length, size=num_snippets)
        for n in range(num_snippets):
            #pick two random demonstrations
            ti, tj = np.random.choice(num_demos, size=(2, ), replace=False)
            #create random snippets
            #find min length of both demos to ensure we can pick a demo no earlier
            #than that chosen in worse preferred demo
            min_length = min(len(demonstrations[ti]), len(demonstrations[tj]))
            if ti < tj:  # pick tj snippet to be later than ti
                ti_start = np.random.randint(min_length - rand_length[n] + 1)
                # print(ti_start, len(demonstrations[tj]))
                tj_start = np.random.randint(ti_start, len(demonstrations[tj]) - rand_length[n] + 1)
            else:  # ti is better so pick later snippet in ti
                tj_start = np.random.randint(min_length - rand_length[n] + 1)
                # print(tj_start, len(demonstrations[ti]))
                ti_start = np.random.randint(tj_start, len(demonstrations[ti]) - rand_length[n] + 1)
            # skip everyother framestack to reduce size
            traj_i = demonstrations[ti][ti_start:ti_start + rand_length[n]:2]
            traj_j = demonstrations[tj][tj_start:tj_start + rand_length[n]:2]

            max_traj_length = max(max_traj_length, len(traj_i), len(traj_j))
            label = int(ti <= tj)
            self.training_obs.append((traj_i, traj_j))
            self.training_labels.append(label)

        self._logger.info(("maximum traj length: {}".format(max_traj_length)))
        return self.training_obs, self.training_labels

    def train(self):
        # check if gpu available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Assume that we are on a CUDA machine, then this should print a CUDA device:
        self._logger.info("device: {}".format(device))
        training_inputs, training_outputs = self.training_obs, self.training_labels
        loss_criterion = nn.CrossEntropyLoss()

        cum_loss = 0.0
        training_data = list(zip(training_inputs, training_outputs))
        for epoch in range(10):  # todo
            np.random.shuffle(training_data)
            training_obs, training_labels = zip(*training_data)
            for i in range(len(training_labels)):

                # traj_i, traj_j has the same length, however, they change as i increases
                traj_i, traj_j = training_obs[i]  # traj_i is a list of array generated by env.step
                traj_i = np.array(traj_i)
                traj_j = np.array(traj_j)
                traj_i = torch.from_numpy(traj_i).float().to(device)
                traj_j = torch.from_numpy(traj_j).float().to(device)

                # training_labels[i] is a boolean integer: 0 or 1
                labels = torch.tensor([training_labels[i]]).to(device)

                # forward + backward + zero out gradient + optimize
                outputs, abs_rewards = self.reward_model.forward(traj_i, traj_j)
                outputs = outputs.unsqueeze(0)
                loss = loss_criterion(outputs, labels) + self.l1_reg * abs_rewards
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                # print stats to see if learning
                item_loss = loss.item()
                cum_loss += item_loss
                if i % 100 == 99:
                    self._logger.info("epoch {}:{} loss {}".format(epoch, i, cum_loss))
                    self._logger.info("abs_returns: {}".format(abs_rewards))
                    cum_loss = 0.0
                    self._logger.info("check pointing")
                    torch.save(self.reward_model.state_dict(), self.reward_path)
        torch.save(self.reward_model.state_dict(), self.reward_path)
        self._logger.info("finished training")
        # print out predicted cumulative returns and actual returns
        #sorted_returns = sorted(self.learning_returns)
        with torch.no_grad():
            pred_returns = [self.predict_traj_return(self.reward_model, traj) for traj in self.pre_expert_data]
        # for i, p in enumerate(pred_returns):
        #     self._logger.info("{} {} {}".format(i, p, sorted_returns[i]))
        info = {
            #"demo_length": [len(d) for d in self.pre_expert_data],
            #"min_snippet_length": self.min_snippet_length,
            #"max_snippet_length": min(np.min([len(d) for d in self.pre_expert_data]), self.max_snippet_length),
            #"len_num_training_obs": len(self.training_obs),
            #"lem_num_labels": len(self.training_labels),
            "accuracy": self.calc_accuracy(self.reward_model, self.training_obs, self.training_labels),
        }
        self._logger.info(
            "accuracy and comparison:\n{}".format('\n'.join(['{}: {}'.format(k, v) for k, v in info.items()]))
        )

    def predict_traj_return(self, net, traj):
        device = self.device
        # torch.set_printoptions(precision=20)
        # torch.use_deterministic_algorithms(True)
        with torch.no_grad():
            rewards_from_obs = net.cum_return(
                torch.from_numpy(np.array(traj)).float().to(device), mode='batch'
            )[0].squeeze().tolist()
            # rewards_from_obs1 = net.cum_return(torch.from_numpy(np.array([traj[0]])).float().to(device))[0].item()
            # different precision
        return sum(rewards_from_obs)  # rewards_from_obs is a list of floats

    def calc_accuracy(self, reward_network, training_inputs, training_outputs):
        #device = self.device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        loss_criterion = nn.CrossEntropyLoss()
        num_correct = 0.
        with torch.no_grad():
            for i in range(len(training_inputs)):
                label = training_outputs[i]
                traj_i, traj_j = training_inputs[i]
                traj_i = np.array(traj_i)
                traj_j = np.array(traj_j)
                traj_i = torch.from_numpy(traj_i).float().to(device)
                traj_j = torch.from_numpy(traj_j).float().to(device)

                #forward to get logits
                outputs, abs_return = reward_network.forward(traj_i, traj_j)
                _, pred_label = torch.max(outputs, 0)
                if pred_label.item() == label:
                    num_correct += 1.
        return num_correct / len(training_inputs)

    def estimate(self, data: list) -> None:
        """
        Overview:
            Estimate reward by rewriting the reward key in each row of the data.
        Arguments:
            - data (:obj:`list`): the list of data used for estimation, with at least \
                 ``obs`` and ``action`` keys.
        Effects:
            - This is a side effect function which updates the reward values in place.
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        res = collect_states(data)
        #res = torch.from_numpy(res)
        res = torch.stack(res).to(device)
        with torch.no_grad():
            sum_rewards, sum_abs_rewards = self.reward_model.cum_return(res, mode='batch')

        for item, rew in zip(data, sum_rewards):  # TODO optimise this loop as well ?
            item['reward'] = rew

    def estimate_observation(self, data: list) -> None:
        """
        Overview:
            Estimate reward by rewriting the reward key in each row of the data.
        Arguments:
            - data (:obj:`list`): the list of data used for estimation, with at least \
                 ``obs`` and ``action`` keys.
        Effects:
            - This is a side effect function which updates the reward values in place.
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        res = collect_states(data)
        res1 = []
        if len(res) == 1:
            res1_ele = torch.from_numpy(res[0])
            res1.append(res1_ele)
        res = res1
        #res = torch.from_numpy(res)
        res = torch.stack(res).to(device)
        with torch.no_grad():
            sum_rewards, sum_abs_rewards = self.reward_model.cum_return(res, mode='batch')

        for item, rew in zip(data, sum_rewards):  # TODO optimise this loop as well ?
            item['reward'] = rew      

    def collect_data(self, data: list) -> None:
        """
        Overview:
            Collecting training data formatted by  ``fn:concat_state_action_pairs``.
        Arguments:
            - data (:obj:`Any`): Raw training data (e.g. some form of states, actions, obs, etc)
        Effects:
            - This is a side effect function which updates the data attribute in ``self``
        """
        pass

    def clear_data(self) -> None:
        """
        Overview:
            Clearing training data. \
            This is a side effect function which clears the data attribute in ``self``
        """
        self.training_obs.clear()
        self.training_labels.clear()

if __name__ == '__main__':
    trex = TrexRewardModelAD()
    #trex._load_expert_data()
    trex._load_expert_folder()
    trex.create_training_data()
    trex.train()