from collections import defaultdict
import os
import numpy as np
from ding.utils.data.collate_fn import default_collate, default_decollate
from easydict import EasyDict
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim import Adam

from core.policy import CILRSPolicy
from core.data import CILRSDataset
from demo.metadrive.hrl_dataset import HRLDataset
from core.policy.hrl_policy.traj_qac import ConvQAC 
from core.policy.hrl_policy.traj_sac import TrajSAC



from ding.envs import BaseEnvManager, SyncSubprocessEnvManager
from ding.config import compile_config
from ding.policy import SACPolicy
from ding.worker import SampleSerialCollector, InteractionSerialEvaluator, BaseLearner, NaiveReplayBuffer
from core.envs import DriveEnvWrapper
#from core.policy.ad_policy.conv_qac import ConvQAC
from core.envs.md_envs.md_hrl_env import MetadriveHrlEnv
from core.policy.hrl_policy.traj_qac import ConvQAC 
from core.policy.hrl_policy.traj_sac import TrajSAC
from core.utils.simulator_utils.evaluator_utils import MetadriveEvaluator
from demo.metadrive.spirl_policy import SPiRLPolicy



TRAJ_CONTROL_MODE = 'acc' # 'acc', 'jerk'
SEQ_TRAJ_LEN = 10
if TRAJ_CONTROL_MODE == 'acc':
    VAE_LOAD_DIR = 'traj_model/seq_len_10_decoder_ckpt'
elif TRAJ_CONTROL_MODE == 'jerk': 
    VAE_LOAD_DIR = 'ckpt_files/new_jerk_decoder_ckpt'
else:
    VAE_LOAD_DIR = None



trex_path = 'test_trex_ad'
trex_reward_folder = trex_path + '/reward_model'
trex_expert_data_folder = trex_path + '/expert_data_folder'
trex_expert_ckpt_path = trex_path + '/expert_macro_policy/april10_iteration40k.pth.tar'
# '/home/SENSETIME/zhoutong/drive_project/ckpt/march23/a1_exp3/iteration_70000.pth.tar'
trex_expert_ckpt_path = '/home/SENSETIME/zhoutong/drive_project/log/april23/hrl_iter20.pth.tar'
trex_expert_ckpt_path = '/home/SENSETIME/zhoutong/drive_project/ckpt/may3/iteration_70000.pth.tar'
metadrive_basic_config = dict(
    exp_name = 'metadrive_basic_sac',
    env=dict(
        metadrive=dict(use_render=True,
            show_seq_traj = True,
            traffic_density = 0.30,
            seq_traj_len = SEQ_TRAJ_LEN,
            traj_control_mode = TRAJ_CONTROL_MODE,
            #map='OSOS', 
            #map='XSXS',
            map='SSSSSSSSSS',
            #show_interface=False,
            use_lateral=True,
            use_speed_reward = True,
            use_heading_reward = True,
            use_jerk_reward = True,
            show_interface=False,
            save_expert_data = False,
            save_expert_traj = True,
            expert_data_folder=trex_expert_data_folder,
            expert_traj_folder=trex_expert_data_folder,
        ),
        manager=dict(
            shared_memory=False,
            max_retry=2,
            context='spawn',
        ),
        n_evaluator_episode=1,
        stop_value=99999,
        collector_env_num=1,
        evaluator_env_num=1,
    ),
    policy=dict(
        cuda=False,
        on_policy=False,
        
        model=dict(
            twin_critic=False,
            obs_shape=[5, 200, 200],
            action_shape=2,
            action_space='reparameterization',
            encoder_hidden_size_list=[128, 128, 64],
            vae_seq_len = SEQ_TRAJ_LEN,
            vae_traj_control_mode = TRAJ_CONTROL_MODE,
            vae_load_dir= VAE_LOAD_DIR, #'/home/SENSETIME/zhoutong/hoffnung/xad/ckpt_files/jerk_ckpt',
        ),
        learn=dict(
            update_per_collect=10,
            batch_size=64,
            learning_rate=3e-4,
            multi_gpu=False,
            init_w = False,
            lr=1e-4,
            epoches=200,
        ),
        collect=dict(
            n_sample=50,
            unroll_len = 1,
        ),
        eval=dict(
            evaluator=dict(
                eval_freq=10,
                )
            ),
        other=dict(
            replay_buffer=dict(
                replay_buffer_size=100000,
            ),
        ), 
        priority= False,
        priority_IS_weight=False,
    )
)

main_config = EasyDict(metadrive_basic_config)


def train(policy, optimizer, loader, tb_logger=None, start_iter=0):
    loss_epoch = defaultdict(list)
    iter_num = start_iter
    policy.reset()

    for data_state, data_traj in tqdm(loader):
        # data_id = '0'
        # data_state['birdview']=data_state['birdview'].squeeze().to(torch.float32)
        # data_state['vehicle_state']=data_state['vehicle_state'].squeeze().to(torch.float32)
        # #data_state['state']
        # data_state = {data_id: data_state}
        
        log_vars = policy.forward(data_state)
        print(log_vars)
    #     optimizer.zero_grad()
    #     total_loss = log_vars['total_loss']
    #     total_loss.backward()
    #     optimizer.step()
    #     log_vars['cur_lr'] = optimizer.defaults['lr']
    #     for k, v in log_vars.items():
    #         loss_epoch[k] += [log_vars[k].item()]
    #         if iter_num % 50 == 0 and tb_logger is not None:
    #             tb_logger.add_scalar("train_iter/" + k, v, iter_num)
    #     iter_num += 1
    # loss_epoch = {k: np.mean(v) for k, v in loss_epoch.items()}
    return iter_num, loss_epoch


def validate(policy, loader, tb_logger=None, epoch=0):
    loss_epoch = defaultdict(list)
    policy.reset()
    for data in tqdm(loader):
        with torch.no_grad():
            log_vars = policy.forward(data)
        for k in list(log_vars.keys()):
            loss_epoch[k] += [log_vars[k]]
    loss_epoch = {k: np.mean(v) for k, v in loss_epoch.items()}
    if tb_logger is not None:
        for k, v in loss_epoch.items():
            tb_logger.add_scalar("validate_epoch/" + k, v, epoch)
    return loss_epoch


def save_ckpt(state, name=None, exp_name=''):
    os.makedirs('checkpoints/' + exp_name, exist_ok=True)
    ckpt_path = 'checkpoints/{}/{}_ckpt.pth'.format(exp_name, name)
    torch.save(state, ckpt_path)


def load_best_ckpt(policy, optimizer=None, root_dir='checkpoints', exp_name='', ckpt_path=None):
    ckpt_dir = os.path.join(root_dir, exp_name)
    assert os.path.isdir(ckpt_dir), ckpt_dir
    files = os.listdir(ckpt_dir)
    assert files, 'No ckpt files found'

    if ckpt_path and ckpt_path in files:
        pass
    elif os.path.exists(os.path.join(ckpt_dir, 'best_ckpt.pth')):
        ckpt_path = 'best_ckpt.pth'
    else:
        ckpt_path = sorted(files)[-1]
    print('Load ckpt:', ckpt_path)
    state_dict = torch.load(os.path.join(ckpt_dir, ckpt_path))
    policy.load_state_dict(state_dict)
    if 'optimizer' in state_dict:
        optimizer.load_state_dict(state_dict['optimizer'])
    epoch = state_dict['epoch']
    iterations = state_dict['iterations']
    best_loss = state_dict['best_loss']
    return epoch, iterations, best_loss


def main(cfg):
    # if cfg.policy.cudnn:
    #     torch.backends.cudnn.benchmark = True
    cfg = compile_config(
        cfg,
        BaseEnvManager,
        SACPolicy,
        BaseLearner,
        SampleSerialCollector,
        InteractionSerialEvaluator,
        NaiveReplayBuffer,
    )
    expert_dir = '/home/SENSETIME/zhoutong/hoffnung/xad/test_trex_ad/expert_data_folder'
    train_dataset = HRLDataset(expert_dir)

    # train_dataset = CILRSDataset(**cfg.data.train)
    # val_dataset = CILRSDataset(**cfg.data.val)
    train_loader = DataLoader(train_dataset, cfg.policy.learn.batch_size, shuffle=True, num_workers=8)
    model = ConvQAC(**cfg.policy.model)
    policy = SPiRLPolicy(cfg.policy, model=model)
    optimizer = Adam(policy._model.parameters(), cfg.policy.learn.lr)
    tb_logger = SummaryWriter('./log/{}/'.format(cfg.exp_name))

    # val_loader = DataLoader(val_dataset, cfg.policy.learn.batch_size, num_workers=8)

    # cilrs_policy = CILRSPolicy(cfg.policy)
    # optimizer = Adam(cilrs_policy._model.parameters(), cfg.policy.learn.lr)
    # tb_logger = SummaryWriter('./log/{}/'.format(cfg.exp_name))
    iterations = 0
    best_loss = 1e8
    start_epoch = 0

    # if cfg.policy.resume:
    #     start_epoch, iterations, best_loss = load_best_ckpt(
    #         cilrs_policy.learn_mode, optimizer, exp_name=cfg.exp_name, ckpt_path=cfg.policy.ckpt_path
    #     )

    for epoch in range(start_epoch, cfg.policy.learn.epoches):
        iter_num, loss = train(policy.learn_mode, optimizer, train_loader, tb_logger, iterations)
        iterations = iter_num
        print('zt1')
        # tqdm.write(
        #     f"Epoch {epoch:03d}, Iter {iter_num:06d}: Total: {loss['total_loss']:2.5f}" +
        #     f" Speed: {loss['speed_loss']:2.5f} Str: {loss['steer_loss']:2.5f}" +
        #     f" Thr: {loss['throttle_loss']:2.5f} Brk: {loss['brake_loss']:2.5f}"
        # )
    #     if epoch % cfg.policy.eval.eval_freq == 0:
    #         loss_dict = validate(cilrs_policy.learn_mode, val_loader, tb_logger, iterations)
    #         total_loss = loss_dict['total_loss']
    #         tqdm.write(f"Validate Total: {total_loss:2.5f}")
    #         state_dict = cilrs_policy.learn_mode.state_dict()
    #         state_dict['optimizer'] = optimizer.state_dict()
    #         state_dict['epoch'] = epoch
    #         state_dict['iterations'] = iterations
    #         state_dict['best_loss'] = best_loss
    #         if total_loss < best_loss and epoch > 0:
    #             tqdm.write("Best Validation Loss!")
    #             best_loss = total_loss
    #             state_dict['best_loss'] = best_loss
    #             save_ckpt(state_dict, 'best', cfg.exp_name)
    #         save_ckpt(state_dict, '{:05d}'.format(epoch), cfg.exp_name)


if __name__ == '__main__':
    main(main_config)
    print('zt')
