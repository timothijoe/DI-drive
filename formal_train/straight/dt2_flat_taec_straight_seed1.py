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
from core.policy.ad_policy.conv_qac import ConvQAC
#from core.envs.md_control_env import MetaDriveControlEnv
#from core.envs.jerk_control_md_env import JerkControlMdEnv
from core.utils.simulator_utils.evaluator_utils import MetadriveEvaluator
#from core.policy.hrl_policy.traj_qac import ConvQAC 
# from core.policy.hrl_policy.control_qac import ControlQAC 
from core.policy.hrl_policy.flat_qac import FlatConvQAC
from core.policy.hrl_policy.traj_sac import TrajSAC
from core.envs.md_traj_env import MetaDriveTrajEnv
from core.policy.hrl_policy.const_qac import ConstQAC 
from core.policy.hrl_policy.traj_sac import TrajSAC

from ding.utils import set_pkg_seed

TRAJ_CONTROL_MODE = 'acc' # 'acc', 'jerk'
VAE_LOAD_DIR='traj_model/99_decoder_ckpt_len1_dim5'
SEQ_TRAJ_LEN = 1
EXP_SEED = 1
metadrive_basic_config = dict(
    exp_name = 'd_nov14_t2_flattaec_straight_seed1',
    seed = EXP_SEED,
    env=dict(
        metadrive=dict(
            show_seq_traj = False,
            traffic_density = 0.3,
            seq_traj_len = SEQ_TRAJ_LEN,
            traj_control_mode = TRAJ_CONTROL_MODE,
            #map='OSOS', 
            #map='XSXS',
            #show_interface=False,
            use_lateral=True,
            use_speed_reward = True,
            use_heading_reward = True,
            use_jerk_reward = True,
            heading_reward=0.15,
            speed_reward = 0.05,
            driving_reward = 0.2,

            crash_vehicle_penalty = 4.0,
            out_of_road_penalty = 5.0,
            ),
        manager=dict(
            shared_memory=False,
            max_retry=2,
            context='spawn',
        ),
        n_evaluator_episode=20, # 20
        stop_value=99999,
        collector_env_num=20, # 20
        evaluator_env_num=10, # 4
    ),
    policy=dict(
        cuda=True,
        model=dict(
            obs_shape=[5, 200, 200],
            action_shape=5,
            vae_latent_dim=5,
            encoder_hidden_size_list=[128, 128, 64],
            vae_seq_len = SEQ_TRAJ_LEN,
            vae_traj_control_mode = TRAJ_CONTROL_MODE,
            vae_load_dir= VAE_LOAD_DIR, #'/home/SENSETIME/zhoutong/hoffnung/xad/ckpt_files/jerk_ckpt',
            steer_rate_constrain_value=0.5,
        ),
        learn=dict(
            update_per_collect=100,
            auto_alpha = False,
            discount_factor = 0.99, # if seq_len = 1, we set 0.99
            batch_size=64,
            learning_rate=3e-4,
            learner=dict(
                hook = dict(save_ckpt_after_iter=5000,),
            ),
        ),
        collect=dict(
            n_sample=5000, # 5000
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

    collector_env.seed(EXP_SEED)
    evaluator_env.seed(EXP_SEED, dynamic_seed=False)
    set_pkg_seed(EXP_SEED, use_cuda=cfg.policy.cuda)

    model = FlatConvQAC(**cfg.policy.model)
    policy = TrajSAC(cfg.policy, model=model)

    tb_logger = SummaryWriter('./log/{}/'.format(cfg.exp_name))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    collector = SampleSerialCollector(cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name)
    evaluator = MetadriveEvaluator(cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name)
    replay_buffer = NaiveReplayBuffer(cfg.policy.other.replay_buffer, tb_logger, exp_name=cfg.exp_name)

    learner.call_hook('before_run')

    while True:
        if evaluator.should_eval(learner.train_iter):
            #stop, rate = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
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
