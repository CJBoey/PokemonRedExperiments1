from os.path import exists
from pathlib import Path
import uuid
from baselines.boey_baselines2.red_gym_env import RedGymEnvV3 as RedGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from baselines.boey_baselines2.custom_network import CustomCombinedExtractorV2
from baselines.boey_baselines2.tensorboard_callback import TensorboardCallback, GammaScheduleCallback
from typing import Callable

def make_env(rank, env_conf, seed=0, es_min_reward_list=None):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env_config['env_id'] = rank
        if es_min_reward_list:
            env_config['early_stopping_min_reward'] = es_min_reward_list[rank]
        env = RedGymEnv(env_conf)
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init

def create_callbacks(use_wandb_logging=False, save_state_dir=None):
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=sess_path,
                                     name_prefix='poke')
    # gamma_schedule_callback = GammaScheduleCallback(init_gamma=0.9996, target_gamma=0.9999, given_timesteps=60_000_000, start_from=9_830_400)
    
    # callbacks = [checkpoint_callback, TensorboardCallback(save_state_dir=save_state_dir), gamma_schedule_callback]
    callbacks = [checkpoint_callback, TensorboardCallback(save_state_dir=save_state_dir)]

    if use_wandb_logging:
        import wandb
        from wandb.integration.sb3 import WandbCallback
        run = wandb.init(
            project="pokemon-train",
            id=sess_id,
            config=env_config,
            sync_tensorboard=True,  
            monitor_gym=True,  
            save_code=True,
        )
        callbacks.append(WandbCallback())
    else:
        run = None
    return callbacks, run

if __name__ == '__main__':

    use_wandb_logging = False
    cpu_multiplier = 0.1  # For R9 7950x: 1.0 for 32 cpu, 1.25 for 40 cpu, 1.5 for 48 cpu
    ep_length = 1024 * 1000 * 30  # 30m steps
    save_freq = 2048 * 10 * 2
    n_steps = int(5120 // cpu_multiplier) * 1
    sess_id = str(uuid.uuid4())[:8]
    sess_path = Path(f'D:/pokered/running/session_{sess_id}_env19_lr3e-4_ent01_bs2048_ep3_5120_vf05_release')
    num_cpu = int(32 * cpu_multiplier)  # Also sets the number of episodes per training iteration
    state_dir = Path(r'D:\pokered\states\env19_release')
    env_config = {
                'headless': True, 'save_final_state': True, 
                'early_stop': True,  # resumed early stopping to ensure reward signal
                'action_freq': 24, 'init_state': 'has_pokedex_nballs_noanim.state', 'max_steps': ep_length, 
                # 'env_max_steps': env_max_steps,
                'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
                'gb_path': 'PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0, 
                'use_screen_explore': False, 'reward_scale': 4, 
                'extra_buttons': False, 'restricted_start_menu': False, 
                'noop_button': True,
                'swap_button': True,
                'enable_item_manager': True,
                'level_reward_badge_scale': 1.0,
                # 'randomize_first_ep_split_cnt': num_cpu,
                # 'start_from_state_dir': state_dir, 
                'save_state_dir': state_dir,
                'explore_weight': 1.5, # 3
                'special_exploration_scale': 1.0,  # double the exploration for special maps (caverns)
                'enable_stage_manager': True,
                'enable_item_purchaser': True,
                'auto_skip_anim': True,
                'auto_skip_anim_frames': 8,
                'early_stopping_min_reward': 2.0,
                'total_envs': num_cpu,
                'level_manager_eval_mode': False,  # True = full run
                # 'randomization': 0.3,
            }
    
    print(env_config)

    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)], start_method='spawn')
    

    learn_steps = 1
    # put a checkpoint here you want to start from
    file_name = r'D:\pokered\running\session_12893b31_env19_lr3e-4_ent01_bs2048_ep3_5120_vf05_v2_es2_fullrun_gamma96_ppo34_1329m\poke_9830400_steps'
    if file_name and not exists(file_name + '.zip'):
        raise Exception(f'File {file_name} does not exist!')
    
    def warmup_schedule(initial_value: float) -> Callable[[float], float]:
        """
        Linear learning rate schedule.

        :param initial_value: Initial learning rate.
        :return: schedule that computes
        current learning rate depending on remaining progress
        """
        def func(progress_remaining: float) -> float:
            """
            Progress will decrease from 1 (beginning) to 0.

            :param progress_remaining:
            :return: current learning rate
            """
            one_update = 0.000125
            n_update = 2
            if progress_remaining > (1 - (one_update * n_update)):  # was warmup for 16 updates 81920 steps, 2.6m total steps.
                return 0.0
            else:
                return initial_value

        return func

    if exists(file_name + '.zip'):
        print(f'\nloading checkpoint: {file_name}')
        new_gamma = 0.9996
        import torch
        policy_kwargs = dict(
            features_extractor_class=CustomCombinedExtractorV2,
            share_features_extractor=True,
            net_arch=[1024, 1024],  # dict(pi=[256, 256], vf=[256, 256])
            activation_fn=torch.nn.ReLU,
        )  # fix for loading checkpoint from other projects
        model = PPO.load(file_name, env=env, ent_coef=0.01, n_epochs=1, gamma=new_gamma, custom_objects=dict(policy_kwargs=policy_kwargs))  # , learning_rate=warmup_schedule(0.0003)
        print(f'Loaded model1 --- LR: {model.learning_rate} OptimizerLR: {model.policy.optimizer.param_groups[0]["lr"]}, ent_coef: {model.ent_coef}, n_epochs: {model.n_epochs}, n_steps: {model.n_steps}, batch_size: {model.batch_size}, gamma: {model.gamma}, rollout_buffer.gamma: {model.rollout_buffer.gamma}')
        model.gamma = new_gamma
        model.rollout_buffer.gamma = new_gamma
        print(model.policy)
        print(f'Loaded model3 --- LR: {model.learning_rate} OptimizerLR: {model.policy.optimizer.param_groups[0]["lr"]}, ent_coef: {model.ent_coef}, gamma: {model.gamma}, rollout_buffer.gamma: {model.rollout_buffer.gamma}')
    else:
        print('\ncreating new model with [512, 512] fully shared layer')
        import torch
        policy_kwargs = dict(
            features_extractor_class=CustomCombinedExtractorV2,
            share_features_extractor=True,
            net_arch=[1024, 1024],  # dict(pi=[256, 256], vf=[256, 256])
            activation_fn=torch.nn.ReLU,
        )
        model = PPO('MultiInputPolicy', env, verbose=1, n_steps=n_steps, batch_size=2048, n_epochs=3, gamma=0.999, tensorboard_log=sess_path,
                    ent_coef=0.01, learning_rate=0.0003, vf_coef=0.5,  # target_kl=0.01,
                    policy_kwargs=policy_kwargs)
         # , policy_kwargs={'net_arch': dict(pi=[1024, 1024], vf=[1024, 1024])}
        
        print(model.policy)

        print(f'start training --- LR: {model.learning_rate} OptimizerLR: {model.policy.optimizer.param_groups[0]["lr"]}, ent_coef: {model.ent_coef}')
    
    callbacks, run = create_callbacks(use_wandb_logging, save_state_dir=state_dir)
    
    for i in range(learn_steps):
        model.learn(total_timesteps=(81_920)*num_cpu*1000*10, callback=CallbackList(callbacks), reset_num_timesteps=True)

        
    if run:
        run.finish()
