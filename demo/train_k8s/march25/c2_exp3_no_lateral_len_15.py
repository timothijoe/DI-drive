import metadrive
from easydict import EasyDict
from functools import partial
from tensorboardX import SummaryWriter

from metadrive import TopDownMetaDrive
from ding.envs import BaseEnvManager, SyncSubprocessEnvManager
from ding.config import compile_config
from ding.policy import SACPolicy
from ding.worker import SampleSerialCollector, InteractionSerialEvaluator, BaseLearner, NaiveReplayBuffer
from core.envs import DriveEnvWrapper
#from core.policy.ad_policy.conv_qac import ConvQAC
from core.envs.md_traj_env import MetaDriveTrajEnv
from core.policy.hrl_policy.traj_qac import ConvQAC 
from core.policy.hrl_policy.traj_sac import TrajSAC
from core.utils.simulator_utils.evaluator_utils import MetadriveEvaluator

TRAJ_CONTROL_MODE = 'acc' # 'acc', 'jerk'
SEQ_TRAJ_LEN = 15

if TRAJ_CONTROL_MODE == 'acc':
    if SEQ_TRAJ_LEN == 10:
        VAE_LOAD_DIR = 'traj_model/seq_len_10_decoder_ckpt'
    elif SEQ_TRAJ_LEN == 15:
        VAE_LOAD_DIR = 'traj_model/seq_len_15_decoder_ckpt'

elif TRAJ_CONTROL_MODE == 'jerk': 
    VAE_LOAD_DIR = 'ckpt_files/new_jerk_decoder_ckpt'
else:
    VAE_LOAD_DIR = None
metadrive_basic_config = dict(
    exp_name = 'c2_exp3_no_lateral_len_15',
    env=dict(
        metadrive=dict(use_render=False,
            show_seq_traj = False,
            traffic_density = 0.3,
            seq_traj_len = SEQ_TRAJ_LEN,
            traj_control_mode = TRAJ_CONTROL_MODE,
            #map='OSOS', 
            #map='XSXS',
            #show_interface=False,
            use_lateral=False,
            use_speed_reward = True,
            use_heading_reward = True,
            use_jerk_reward = True,
            heading_reward=0.2,
        ),
        manager=dict(
            shared_memory=False,
            max_retry=2,
            context='spawn',
        ),
        n_evaluator_episode=12,
        stop_value=99999,
        collector_env_num=20,
        evaluator_env_num=4,
    ),
    policy=dict(
        cuda=True,
        model=dict(
            obs_shape=[5, 200, 200],
            action_shape=2,
            encoder_hidden_size_list=[128, 128, 64],
            vae_seq_len = SEQ_TRAJ_LEN,
            vae_traj_control_mode = TRAJ_CONTROL_MODE,
            vae_load_dir= VAE_LOAD_DIR, #'/home/SENSETIME/zhoutong/hoffnung/xad/ckpt_files/jerk_ckpt',
        ),
        learn=dict(
            update_per_collect=100,
            batch_size=64,
            learning_rate=3e-4,
        ),
        collect=dict(
            n_sample=5000,
        ),
        eval=dict(
            evaluator=dict(
                eval_freq=1000,
                )
            ),
        other=dict(
            replay_buffer=dict(
                replay_buffer_size=100000,
            ),
        ), 
    )
)

main_config = EasyDict(metadrive_basic_config)


def wrapped_env(env_cfg, wrapper_cfg=None):
    return DriveEnvWrapper(MetaDriveTrajEnv(config=env_cfg), wrapper_cfg)


def main(cfg):
    cfg = compile_config(
        cfg,
        SyncSubprocessEnvManager,
        SACPolicy,
        BaseLearner,
        SampleSerialCollector,
        InteractionSerialEvaluator,
        NaiveReplayBuffer,
    )

    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    collector_env = SyncSubprocessEnvManager(
        env_fn=[partial(wrapped_env, cfg.env.metadrive) for _ in range(collector_env_num)],
        cfg=cfg.env.manager,
    )
    evaluator_env = SyncSubprocessEnvManager(
        env_fn=[partial(wrapped_env, cfg.env.metadrive) for _ in range(evaluator_env_num)],
        cfg=cfg.env.manager,
    )

    model = ConvQAC(**cfg.policy.model)
    policy = TrajSAC(cfg.policy, model=model)

    tb_logger = SummaryWriter('./log/{}/'.format(cfg.exp_name))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    collector = SampleSerialCollector(cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name)
    evaluator = MetadriveEvaluator(cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name)
    replay_buffer = NaiveReplayBuffer(cfg.policy.other.replay_buffer, tb_logger, exp_name=cfg.exp_name)

    learner.call_hook('before_run')

    while True:
        if evaluator.should_eval(learner.train_iter):
            # stop, rate = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            stop, rate = evaluator.evall(learner.save_checkpoint, learner.train_iter, collector.envstep, collector._total_episode_count, collector._total_duration)
            if stop:
                break
        # Sampling data from environments
        new_data = collector.collect(cfg.policy.collect.n_sample, train_iter=learner.train_iter)
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        for i in range(cfg.policy.learn.update_per_collect):
            train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
            if train_data is None:
                break
            learner.train(train_data, collector.envstep)
    learner.call_hook('after_run')

    collector.close()
    evaluator.close()
    learner.close()

if __name__ == '__main__':
    main(main_config)