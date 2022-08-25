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
from ding.utils.data import offline_data_save_type, default_collate
from ding.utils import build_logger
from dizoo.atari.envs.atari_wrappers import wrap_deepmind
from dizoo.mujoco.envs.mujoco_wrappers import wrap_mujoco
from ding.torch_utils import to_tensor

from ding.reward_model.base_reward_model import BaseRewardModel
from ding.reward_model.rnd_reward_model import collect_states 
import os 


# assert os.path.exists(trex_path)
# if not os.path.exists(trex_reward_folder):
#     os.makedirs(trex_reward_folder)

# trex_reward_folder = trex_path + '/reward_model'
# trex_ckpt_name = trex_reward_folder + '/reward_ckpt_sigmoid_drex_20state'

config = dict(
    dataset_path = '/test_drex',
    noise_level = ['1.0','0.9','0.8','0.7','0.6','0.5','0.4','0.3','0.2','0.1','0.0'],
    drex_path = '/test_drex',
    reward_model_name = 'drex_reward_model',
)


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
        self.state_hidden_size = 256
        self.image_hidden_size = 256
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

        self.image_encoder = nn.Sequential(
            nn.Linear(flatten_size,self.image_hidden_size),
            self.act,
        )
        self.mid = nn.Sequential(
            nn.Linear(self.image_hidden_size+self.state_hidden_size, hidden_size_list[-2]),
            self.act,
            nn.Linear(hidden_size_list[-2], hidden_size_list[-1]),
            nn.Sigmoid(),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(2,self.state_hidden_size),
            self.act,
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
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        birdview = x['obs']['birdview'].float().to(device)
        vehicle_state = x['obs']['vehicle_state'].float().to(device)
        vehicle_state = vehicle_state.transpose(0,1)
        speed = vehicle_state[3]
        steer = vehicle_state[5]
        speed = speed.reshape(-1,1)
        steer = steer.reshape(-1,1)
        birdview_embedding = self.main(birdview)
        birdview_embedding = self.image_encoder(birdview_embedding)
        state_embedding = self.state_encoder(torch.cat((speed,steer),dim=1))
        x = torch.cat((birdview_embedding,state_embedding),dim=1)
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
        #r = torch.tanh(r)
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
        self.reward_model = TrexModel((5,200,200))
        self.pre_expert_data = []
        self.train_data = []
        self.expert_data_loader = None
        self.opt = optim.Adam(self.reward_model.parameters(), 1e-4)#1e-3
        self.train_iter = 0
        self.learning_returns = []
        self.learning_rewards = []
        self.training_obs = []
        self.training_labels = []
        self.testing_obs = []
        self.testing_labels = []
        # self.num_trajs = self.cfg.reward_model.num_trajs
        # self.num_snippets = self.cfg.reward_model.num_snippets
        # minimum number of short subtrajectories to sample
        self.min_snippet_length = 15
        # maximum number of short subtrajectories to sample
        self.max_snippet_length = 50
        #self.fixed_snippet_length = 15
        self.l1_reg = 0
        self.data_for_save = {}
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.reward_model.to(device)
        self.demonstrations = []
        #self.load_expert_data()
    
    def _load_expert_folder(self,pickle_folder) -> None:
        import os
        expert_data_list = []
        file_list = []
        for cur_file in os.listdir(pickle_folder):
            cur_file_dir = os.path.join(pickle_folder, cur_file)
            file_list.append(cur_file_dir)
        # return file_list 
        # sort by filename
        file_list.sort()
        for file in file_list:
            with open(file, 'rb') as f:
                episode_data = pickle.load(f)
                expert_data_list.append(episode_data)
        print('zt')
        print(len(expert_data_list))
        #demonstrations = sorted(self.expert_data_list, key = lambda x: x["episode_rwd"], reverse=False)#False
        demonstrations = expert_data_list
        for demo in demonstrations:
            damo = []
            for transition in demo['transition_list']:
                transition['obs']['birdview'] = transition['obs']['birdview'].transpose((2, 0, 1))
                transition['next_obs']['birdview'] = transition['next_obs']['birdview'].transpose((2, 0, 1))
                damo.append(transition)
            self.demonstrations.append(damo)
#        self.demonstrations.append(damos) 

    
    def create_training_data(self,drex_path):
        self._logger, self._tblogger = build_logger(
            path='./{}/log/{}'.format(drex_path, 'drex_reward_model'), name='drex_reward_model'
        )
        #demonstrations = self.pre_expert_data
        demonstrations = self.demonstrations 
        num_trajs = len(demonstrations)
        num_snippets = 6000 #600/6000 TODO
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

        #fixed size snippets with progress prior
        rand_length = np.random.randint(min_snippet_length, max_snippet_length, size=num_snippets)
        num_level = 11 #TODO
        every_demo_in_level = 20
        for n in range(num_snippets):
            #pick two random demonstrations
            level_i, level_j = np.random.choice(num_level, size=(2, ), replace=False)
            index_i, index_j = np.random.choice(every_demo_in_level-1, size=(2, ), replace=True)
            ti = level_i * every_demo_in_level + index_i
            tj = level_j * every_demo_in_level + index_j
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

        for n in range(num_snippets//20):
            #pick two random demonstrations
            level_i, level_j = np.random.choice(num_level, size=(2, ), replace=False)
            index_i, index_j = every_demo_in_level-1,every_demo_in_level-1
            ti = level_i * every_demo_in_level + index_i
            tj = level_j * every_demo_in_level + index_j
            #     ti, tj = np.random.choice(num_demos, size=(2, ), replace=False)
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
            self.testing_obs.append((traj_i, traj_j))
            self.testing_labels.append(label)

        self._logger.info(("maximum traj length: {}".format(max_traj_length)))
        return self.training_obs,self.training_labels,self.testing_obs, self.testing_labels

    def train(self,drex_path,model_name):
        self.reward_path = drex_path + '/' + model_name
        # check if gpu available
        best_acc = 0.0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Assume that we are on a CUDA machine, then this should print a CUDA device:
        self._logger.info("device: {}".format(device))
        #split dataset
        dataset = list(zip(self.training_obs, self.training_labels))
        np.random.shuffle(dataset)
        loss_criterion = nn.CrossEntropyLoss()
        cum_loss = 0.0
        training_data = list(zip(self.training_obs, self.training_labels))
        training_data = to_tensor(training_data)
        for epoch in range(10):  # todo
            np.random.shuffle(training_data)
            training_obs, training_labels = zip(*training_data)
            for i in range(len(training_labels)):
                # traj_i, traj_j has the same length, however, they change as i increases
                traj_i, traj_j = training_obs[i]  # traj_i is a list of array generated by env.step
                # traj_i = np.array(traj_i)
                # traj_j = np.array(traj_j)
                data_i = {}
                data_j = {}
                for key in traj_i[0].keys():
                    list_i = []
                    list_j = []
                    for _i in range(len(traj_i)):
                        list_j.append(traj_j[_i][key])
                        list_i.append(traj_i[_i][key])
                    d_i = default_collate(list_i)
                    d_j = default_collate(list_j)
                    data_i[key] = d_i
                    data_j[key] = d_j

                # training_labels[i] is a boolean integer: 0 or 1
                labels = torch.tensor([training_labels[i]]).to(device)

                outputs, abs_rewards = self.reward_model.forward(data_i, data_j)
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
                    #torch.save(self.reward_model.state_dict(), self.reward_path)
            accuracy_test = self.calc_accuracy(self.reward_model, self.testing_obs, self.testing_labels)
            info = {
            "accuracy_train": self.calc_accuracy(self.reward_model, self.training_obs, self.training_labels),
            "accuracy_test": accuracy_test,
            }
            self._logger.info(
            "accuracy and comparison:\n{}".format('\n'.join(['{}: {}'.format(k, v) for k, v in info.items()]))
            )
            if accuracy_test > best_acc:
                torch.save(self.reward_model.state_dict(), self.reward_path)
        self._logger.info("finished training")
        # print out predicted cumulative returns and actual returns
        #sorted_returns = sorted(self.learning_returns)
        with torch.no_grad():
            pred_returns = [self.predict_traj_return(self.reward_model, traj) for traj in self.pre_expert_data]
        # for i, p in enumerate(pred_returns):
        #     self._logger.info("{} {} {}".format(i, p, sorted_returns[i]))
        info = {
            "accuracy_train": self.calc_accuracy(self.reward_model, self.training_obs, self.training_labels),
            "accuracy_test": self.calc_accuracy(self.reward_model, self.testing_obs, self.testing_labels),
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
                data_i = {}
                data_j = {}
                for key in traj_i[0].keys():
                    list_i = []
                    list_j = []
                    for i in range(len(traj_i)):
                        list_j.append(traj_j[i][key])
                        list_i.append(traj_i[i][key])
                    d_i = default_collate(list_i)
                    d_j = default_collate(list_j)
                    data_i[key] = d_i
                    data_j[key] = d_j
                #forward to get logits
                outputs, abs_return = reward_network.forward(data_i, data_j)
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
        train_data_augmented = self.reward_deepcopy(data)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        data_in = {}
        output = {}
        output['obs'] = {}
        for key in train_data_augmented[0]['obs']:
            list_i = []
            for i in range(len(train_data_augmented)):
                list_i.append(train_data_augmented[i]['obs'][key])
            d_i = default_collate(list_i)
            data_in[key] = d_i
        #data_in['state'] = data_in.pop('birdview')
        output['obs']['birdview'] = data_in['birdview']
        output['obs']['vehicle_state'] = data_in['vehicle_state']
        with torch.no_grad():
            sum_rewards, sum_abs_rewards = self.reward_model.cum_return(output, mode='batch')

        for item, rew in zip(train_data_augmented, sum_rewards):  # TODO optimise this loop as well ?
            #item['reward'] = rew
            #add final reward 
            if item['reward'] > 5.0 :
                item['reward'] = rew + 100
            # elif item['reward'] < -3.0 :
            #     item['reward'] = rew - 50
            else:
                item['reward'] = rew

        return train_data_augmented


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

def train(config):
    noise_level = config['noise_level']
    model_name = config['reward_model_name']
    drex_folder_path = config['drex_path']
    dataset_path = config['dataset_path']
    drex = TrexRewardModelAD()
    for i in range(len(noise_level)):
        name = noise_level[i]
        drex_expert_data_folder = dataset_path + '/noise' + name 
        drex._load_expert_folder(drex_expert_data_folder)
    drex.create_training_data(drex_folder_path)
    drex.train(drex_folder_path,model_name)



if __name__ == '__main__':
    train(config)