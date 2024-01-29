
import sys
from typing import Union
import uuid 
import os
from math import floor, sqrt
import json
import pickle
from pathlib import Path

import copy
import random
import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt
from skimage.transform import resize
from pyboy import PyBoy
# import hnswlib
import mediapy as media
import pandas as pd
import math
import datetime

from gymnasium import Env, spaces
from pyboy.utils import WindowEvent
from baselines.boey_baselines2.constants import GYM_INFO, SPECIAL_MAP_IDS, IGNORED_EVENT_IDS, SPECIAL_KEY_ITEM_IDS, \
    ALL_KEY_ITEMS, ALL_HM_IDS, ALL_POKEBALL_IDS, ALL_HEALABLE_ITEM_IDS, ALL_GOOD_ITEMS, GOOD_ITEMS_PRIORITY, \
    POKEBALL_PRIORITY, POTION_PRIORITY, REVIVE_PRIORITY, STATES_TO_SAVE_LOAD, LEVELS
from baselines.boey_baselines2.pokered_constants import MAP_DICT, MAP_ID_REF, WARP_DICT, WARP_ID_DICT, BASE_STATS, \
    SPECIES_TO_ID, ID_TO_SPECIES, CHARMAP, MOVES_INFO_DICT, MART_MAP_IDS, MART_ITEMS_ID_DICT, ITEM_TM_IDS_PRICES
from baselines.boey_baselines2.ram_addresses import RamAddress as RAM
from baselines.boey_baselines2.stage_manager import StageManager, STAGE_DICT, POKECENTER_TO_INDEX_DICT
from skimage.transform import downscale_local_mean

class RedGymEnvV3(Env):


    def __init__(
        self, config=None):

        self.debug = config['debug']
        self.s_path = config['session_path']
        self.save_final_state = config['save_final_state']
        self.print_rewards = config['print_rewards']
        self.vec_dim = 4320 #1000
        self.headless = config['headless']
        self.num_elements = 20000 # max
        self.init_state = config['init_state']
        self.act_freq = config['action_freq']
        self.max_steps = config['max_steps']
        self.early_stopping = config['early_stop']
        self.early_stopping_min_reward = 2.0 if 'early_stopping_min_reward' not in config else config['early_stopping_min_reward']
        self.save_video = config['save_video']
        self.fast_video = config['fast_video']
        self.video_interval = 256 * self.act_freq
        self.downsample_factor = 2
        self.frame_stacks = 3
        self.explore_weight = 1 if 'explore_weight' not in config else config['explore_weight']
        self.use_screen_explore = True if 'use_screen_explore' not in config else config['use_screen_explore']
        self.randomize_first_ep_split_cnt = 0 if 'randomize_first_ep_split_cnt' not in config else config['randomize_first_ep_split_cnt']
        self.similar_frame_dist = config['sim_frame_dist']
        self.reward_scale = 1 if 'reward_scale' not in config else config['reward_scale']
        self.extra_buttons = False if 'extra_buttons' not in config else config['extra_buttons']
        self.noop_button = False if 'noop_button' not in config else config['noop_button']
        self.swap_button = True if 'swap_button' not in config else config['swap_button']
        self.restricted_start_menu = False if 'restricted_start_menu' not in config else config['restricted_start_menu']
        self.level_reward_badge_scale = 0 if 'level_reward_badge_scale' not in config else config['level_reward_badge_scale']
        self.instance_id = str(uuid.uuid4())[:8] if 'instance_id' not in config else config['instance_id']
        self.start_from_state_dir = None if 'start_from_state_dir' not in config else config['start_from_state_dir']
        self.save_state_dir = None if 'save_state_dir' not in config else config['save_state_dir']
        self.randomization = 0 if 'randomization' not in config else config['randomization']
        self.special_exploration_scale = 0 if 'special_exploration_scale' not in config else config['special_exploration_scale']
        self.enable_item_manager = False if 'enable_item_manager' not in config else config['enable_item_manager']
        self.enable_stage_manager = False if 'enable_stage_manager' not in config else config['enable_stage_manager']
        self.enable_item_purchaser = False if 'enable_item_purchaser' not in config else config['enable_item_purchaser']
        self.auto_skip_anim = False if 'auto_skip_anim' not in config else config['auto_skip_anim']
        self.auto_skip_anim_frames = 8 if 'auto_skip_anim_frames' not in config else config['auto_skip_anim_frames']
        self.env_id = str(random.randint(1, 9999)).zfill(4) if 'env_id' not in config else config['env_id']
        self.env_max_steps = [] if 'env_max_steps' not in config else config['env_max_steps']
        self.total_envs = 48 if 'total_envs' not in config else config['total_envs']
        self.level_manager_eval_mode = False if 'level_manager_eval_mode' not in config else config['level_manager_eval_mode']
        self.s_path.mkdir(exist_ok=True)
        self.warmed_up = False  # for randomize_first_ep_split_cnt usage
        self.reset_count = 0
        self.all_runs = []
        self.n_pokemon_features = 23
        self.gym_info = GYM_INFO
        self._last_episode_stats = None
        self.print_debug = False

        if self.max_steps is None:
            assert self.env_max_steps, 'max_steps and env_max_steps cannot be both None'
            if self.env_id < len(self.env_max_steps):
                self.max_steps = self.env_max_steps[self.env_id]
            else:
                self.max_steps = self.env_max_steps[-1]  # use last env_max_steps
                print(f'Warning env_id {self.env_id} is out of range, using last env_max_steps: {self.max_steps}')

        # Set this in SOME subclasses
        self.metadata = {"render.modes": []}
        self.reward_range = (0, 15000)
        # self.pokecenter_ids = [0x29, 0x3A, 0x40, 0x44, 0x51, 0x59, 0x85, 0x8D, 0x9A, 0xAB, 0xB6]
        self.pokecenter_ids = [0x01, 0x02, 0x03, 0x0F, 0x15, 0x05, 0x06, 0x04, 0x07, 0x08, 0x0A, 0x09]
        self.early_done = False
        self.current_level = 0
        self.level_manager_initialized = False
        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
        ]
        
        if self.extra_buttons:
            self.valid_actions.extend([
                WindowEvent.PRESS_BUTTON_START,
                # WindowEvent.PASS
            ])

        if self.noop_button:
            self.valid_actions.extend([
                WindowEvent.PASS
            ])
        
        if self.swap_button:
            self.valid_actions.extend([
                988,  # 988 is special SWAP PARTY action
            ])

        self.release_arrow = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP
        ]

        self.release_button = [
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B
        ]

        self.noop_button_index = self.valid_actions.index(WindowEvent.PASS)
        self.swap_button_index = self.valid_actions.index(988)
        self.output_shape = (144//2, 160//2)
        self.mem_padding = 2
        self.memory_height = 8
        self.col_steps = 16
        self.output_full = (
            self.frame_stacks,
            self.output_shape[0],
            self.output_shape[1]
        )
        self.output_vector_shape = (99, )

        # Set these in ALL subclasses
        self.action_space = spaces.Discrete(len(self.valid_actions))
        # self.observation_space = spaces.Box(low=0, high=255, shape=self.output_full, dtype=np.uint8)
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=255, shape=self.output_full, dtype=np.uint8),
            'minimap': spaces.Box(low=0, high=1, shape=(14, 9, 10), dtype=np.float32),
            'minimap_sprite': spaces.Box(low=0, high=390, shape=(9, 10), dtype=np.int16),
            'minimap_warp': spaces.Box(low=0, high=830, shape=(9, 10), dtype=np.int16),
            'vector': spaces.Box(low=-1, high=1, shape=self.output_vector_shape, dtype=np.float32),
            'map_ids': spaces.Box(low=0, high=255, shape=(10,), dtype=np.uint8),
            'map_step_since': spaces.Box(low=-1, high=1, shape=(10, 1), dtype=np.float32),
            'item_ids': spaces.Box(low=0, high=255, shape=(20,), dtype=np.uint8),
            'item_quantity': spaces.Box(low=-1, high=1, shape=(20, 1), dtype=np.float32),
            'poke_ids': spaces.Box(low=0, high=255, shape=(12,), dtype=np.uint8),
            'poke_type_ids': spaces.Box(low=0, high=255, shape=(12, 2), dtype=np.uint8),
            'poke_move_ids': spaces.Box(low=0, high=255, shape=(12, 4), dtype=np.uint8),
            'poke_move_pps': spaces.Box(low=0, high=1, shape=(12, 4, 2), dtype=np.float32),
            'poke_all': spaces.Box(low=0, high=1, shape=(12, self.n_pokemon_features), dtype=np.float32),
            'event_ids': spaces.Box(low=0, high=2570, shape=(128,), dtype=np.int16),
            'event_step_since': spaces.Box(low=-1, high=1, shape=(128, 1), dtype=np.float32),
            # 'in_battle_mask': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        })

        head = 'headless' if config['headless'] else 'SDL2'

        self.pyboy = PyBoy(
                config['gb_path'],
                debugging=False,
                disable_input=False,
                window_type=head,
                hide_window='--quiet' in sys.argv,
                game_wrapper=True,
            )

        self.screen = self.pyboy.botsupport_manager().screen()
        self.wrapper = self.pyboy.game_wrapper()

        if not config['headless']:
            self.pyboy.set_emulation_speed(1)
            
        self.reset()

    def reset(self, seed=None, options=None):
        self.seed = seed
        
        # if self.use_screen_explore:
        #     self.init_knn()
        # else:
        self.init_map_mem()
        self.init_caches()
        self.level_completed = False
        # self.level_completed_skip_type = None
        self.previous_level = self.current_level
        self.current_level = 0
        self.secret_switch_states = {}
        self.stuck_cnt = 0
        self.elite_4_lost = False
        self.elite_4_early_done = False
        self.elite_4_started_step = None

        # fine tuning, disable level manager for now
        if self.save_state_dir is not None:
            all_level_dirs = list(self.save_state_dir.glob('level_*'))
            # print(f'all_level_dirs: {all_level_dirs}')
            # print(f'LEVELS len {len(LEVELS)}')

            # level 0 = clean states

            highest_level = 0
            # oldest_date_created = datetime.datetime.now()  # oldest date created state folder across levels
            # stale_level = 0
            MIN_CLEAR = 5  # minimum states to have in the level to be considered cleared
            for level_dir in all_level_dirs:
                try:
                    level = int(level_dir.name.split('_')[-1])
                except:
                    continue
                if level >= len(LEVELS):
                    continue
                # if level > highest_level:
                level_states_ordered = sorted(list(level_dir.glob('*')), key=os.path.getmtime)
                num_states = len(level_states_ordered)
                if num_states >= MIN_CLEAR:
                    if level > highest_level:
                        highest_level = level
                    # if level < len(LEVELS):
                    #     # look for stalest level
                    #     # do not consider ended game level
                    #     level_newest_date_created = datetime.datetime.fromtimestamp(os.path.getmtime(level_states_ordered[-1]))
                    #     if level_newest_date_created < oldest_date_created:
                    #         oldest_date_created = level_newest_date_created
                    #         stale_level = level - 1
            explored_levels = highest_level + 1
            # is_assist_env = False
            if explored_levels == 1:
                # only level 0
                # all envs in charge of level 0
                self.level_in_charge = 0
            else:
                split_percent = 0.5
                n_level_env_count = math.ceil(self.total_envs * split_percent / explored_levels)
                total_level_env_count = n_level_env_count * explored_levels
                # except level 4, level 4 only required 1 env
                level_4_env_ids = [i for i in range(4 * n_level_env_count, 5 * n_level_env_count)]
                # total_assist_env = self.total_envs - total_level_env_count
                if self.env_id < total_level_env_count and self.env_id not in level_4_env_ids[1:]:
                    # level env
                    # eg: explored_levels 2
                    # env_id 0 to 11, level_in_charge 0
                    # env_id 12 to 23, level_in_charge 1
                    self.level_in_charge = self.env_id // n_level_env_count
                    print(f'env_id: {self.env_id}, level: {self.level_in_charge}, level_env')
                else:
                    # assist env
                    # check stats of level envs in save_state_dir / level_{level_in_charge}.txt
                    # content of file is something like: SSSSFFS
                    # S: success, F: failed
                    # get the last 20 characters, count the number of S and F
                    # assign at failure rate for each level env
                    # the level env with highest failure rate will more likely to be assigned to assist env
                    level_stats = {}
                    for level in range(explored_levels):
                        level_stats[level] = {'S': 0, 'F': 0}
                        stats_file = self.save_state_dir / Path('stats')
                        level_file = stats_file / Path(f'level_{level}.txt')
                        if stats_file.exists() and level_file.exists():
                            with open(level_file, 'r') as f:
                                stats = f.read()
                                # make sure have atleast 10 stats
                                if len(stats) < 5:
                                    continue
                                for char in stats[-10:]:
                                    level_stats[level][char] += 1
                    
                    # calculate failure rate
                    for level in range(explored_levels):
                        if level_stats[level]['S'] + level_stats[level]['F'] == 0:
                            # insufficient stats, assign failure rate to 1 first
                            level_stats[level]['failure_rate'] = 1
                        else:
                            level_stats[level]['failure_rate'] = level_stats[level]['F'] / (level_stats[level]['S'] + level_stats[level]['F'])
                    total_failure_rate = sum([level_stats[level]['failure_rate'] for level in range(explored_levels)])
                    level_selection_chance_list = [level_stats[level]['failure_rate'] / total_failure_rate for level in range(explored_levels)]
                    # select level based on chance with np.random.choice
                    # if no stats, equal chance to select any level
                    self.level_in_charge = np.random.choice(explored_levels, p=level_selection_chance_list)
                    print(f'env_id: {self.env_id}, level: {self.level_in_charge}, assist_env chance: {[f"{level}: {level_selection_chance_list[level]:.2f}" for level in range(explored_levels)]}')

            self.current_level = self.level_in_charge
            # self.current_level = 5
            # is_end_game = highest_level == len(LEVELS)
            # print(f'highest_level: {highest_level}, stale_level: {stale_level}')
            # if self.early_done:
            #     # check if is highest level
            #     if self.previous_level == highest_level and not is_end_game:
            #         # 10% chance to start from oldest state file level
            #         if np.random.rand() < 0.1:
            #             # start from stale level
            #             self.current_level = stale_level
            #             print(f'earlydone, HL, start from stale_level: {stale_level}')
            #         else:
            #             # start from highest_level
            #             self.current_level = highest_level
            #             print(f'earlydone, HL, start from highest_level: {highest_level}')
            #     else:
            #         # non-highest level
            #         # if early stoppped, restart from the same level
            #         # to ensure that the agent can complete the level
            #         self.current_level = self.previous_level
            #         print(f'earlydone, NHL, start from same level: {self.previous_level}')
            # else:
            #     # level completed
            #     if not is_end_game:
            #         if not self.level_manager_initialized:
            #             # level manager init
            #             # still trying to complete the game
            #             # 0.1 chance to start from any level before highest_level (for each level before highest level)
            #             # the remaining chance to start from highest_level
            #             if np.random.rand() < 0.1 * highest_level:
            #                 # start from any level before highest_level
            #                 self.current_level = np.random.randint(0, highest_level)
            #             else:
            #                 # start from highest_level
            #                 self.current_level = highest_level
            #             # initialized at step()
            #             # self.level_manager_initialized = True
            #         else:
            #             # 10% chance to stay at the same level
            #             if np.random.rand() < 0.1:
            #                 # stay at the same level
            #                 self.current_level = self.previous_level
            #             else:
            #                 # start from highest_level
            #                 self.current_level = highest_level
            #     else:
            #         # game completed
            #         # equal chance to start from any level
            #         if not self.level_manager_initialized:
            #             self.current_level = np.random.randint(0, highest_level)
            #             # initialized at step()
            #             # self.level_manager_initialized = True
            #         else:
            #             # 90% chance to stay at the same level
            #             if np.random.rand() < 0.9:
            #                 # stay at the same level
            #                 self.current_level = self.previous_level
            #             else:
            #                 # start from stale_level
            #                 self.current_level = stale_level
            # print(f'starting from level: {self.current_level}')
            if self.current_level == 0:
                pass
            else:
                # select all_state_dirs from current_level
                all_state_dirs = list((self.save_state_dir / Path(f'level_{self.current_level}')).glob('*'))

                # select N newest folders by using os,path.getmtime
                selected_state_dirs = sorted(all_state_dirs, key=os.path.getmtime)[-MIN_CLEAR:]  # using MIN_CLEAR for now
                
                if len(selected_state_dirs) == 0:
                    raise ValueError('start_from_state_dir is empty')
                # load the state randomly from the directory
                state_dir = np.random.choice(selected_state_dirs)
                print(f'env_id: {self.env_id}, load state {state_dir}, level: {self.current_level}')
                self.load_state(state_dir)
        if self.current_level == 0:
            print(f'env_id: {self.env_id}, level: {self.current_level}')
            state_to_init = self.init_state
            if self.randomization:
                assert isinstance(self.randomization, float)
                if np.random.rand() < self.randomization:
                    randomization_state_dir = 'randomization_states'
                    state_list = list(Path(randomization_state_dir).glob('*.state'))
                    if state_list:
                        state_to_init = np.random.choice(state_list)
            # restart game, skipping credits
            with open(state_to_init, "rb") as f:
                self.pyboy.load_state(f)
            
            self.recent_frames = np.zeros(
                (self.frame_stacks, self.output_shape[0], 
                self.output_shape[1]),
                dtype=np.uint8)

            self.agent_stats = []
            self.base_explore = 0
            self.max_opponent_level = 0
            self.max_event_rew = 0
            self.max_level_rew = 0
            self.party_level_base = 0
            self.party_level_post = 0
            self.last_health = 1
            self.last_num_poke = 1
            self.last_num_mon_in_box = 0
            self.total_healing_rew = 0
            self.died_count = 0
            self.prev_knn_rew = 0
            self.visited_pokecenter_list = []
            self.last_10_map_ids = np.zeros((10, 2), dtype=np.float32)
            self.last_10_coords = np.zeros((10, 2), dtype=np.uint8)
            self.past_events_string = ''
            self.last_10_event_ids = np.zeros((128, 2), dtype=np.float32)
            self.early_done = False
            self.step_count = 0
            self.past_rewards = np.zeros(10240, dtype=np.float32)
            self.base_event_flags = self.get_base_event_flags()
            assert len(self.all_events_string) == 2552, f'len(self.all_events_string): {len(self.all_events_string)}'
            self.rewarded_events_string = '0' * 2552
            self.seen_map_dict = {}
            self.update_last_10_map_ids()
            self.update_last_10_coords()
            self.update_seen_map_dict()
            self._cut_badge = False
            self._have_hm01 = False
            self._can_use_cut = False
            self._surf_badge = False
            self._have_hm03 = False
            self._can_use_surf = False
            self._have_pokeflute = False
            self._have_silph_scope = False
            self.used_cut_coords_dict = {}
            self._last_item_count = 0
            self._is_box_mon_higher_level = False
            self.secret_switch_states = {}
            self.hideout_elevator_maps = []
            self.use_mart_count = 0
            self.use_pc_swap_count = 0
            if self.enable_stage_manager:
                self.stage_manager = StageManager()
            # self._replace_ss_ticket_w_
            self.progress_reward = self.get_game_state_reward()
            self.total_reward = sum([val for _, val in self.progress_reward.items()])
            self.reset_count += 1
        self.early_done = False
        
        if self.save_video:
            base_dir = self.s_path / Path('rollouts')
            base_dir.mkdir(exist_ok=True)
            full_name = Path(f'full_reset_{self.reset_count}_id{self.instance_id}').with_suffix('.mp4')
            # model_name = Path(f'model_reset_{self.reset_count}_id{self.instance_id}').with_suffix('.mp4')
            self.full_frame_writer = media.VideoWriter(base_dir / full_name, (144, 160), fps=60)
            self.full_frame_writer.__enter__()
            self.full_frame_write_full_path = base_dir / full_name
            # self.model_frame_writer = media.VideoWriter(base_dir / model_name, self.output_full[:2], fps=60)
            # self.model_frame_writer.__enter__()
       
        return self.render(), {}
    
    def get_highest_reward_state_dir_based_on_reset_count(self, dirs_given, weightage=0.01):
        '''
        path_given is all_state_dirs
        all_state_dirs is self.start_from_state_dir.glob('*')
        state folders name as such '{self.total_reward:5.2f}_{session_id}_{self.instance_id}_{self.reset_count}'
        return the state folder with highest total_reward divided by reset_count.
        '''
        if not dirs_given:
            return None
        if weightage <= 0:
            print(f'weightage should be greater than 0, weightage: {weightage}')
            weightage = 0.01
        dirs_given = list(dirs_given)
        # get highest total_reward divided by reset_count
        return max(dirs_given, key=lambda x: float(x.name.split('_')[0]) / (float(x.name.split('_')[-1]) + weightage))
    
    def debug_save(self, is_failed=True):
        return self.save_all_states(is_failed=is_failed)
    
    def save_all_states(self, is_failed=False):
        # STATES_TO_SAVE_LOAD = ['recent_frames', 'agent_stats', 'base_explore', 'max_opponent_level', 'max_event_rew', 'max_level_rew', 'last_health', 'last_num_poke', 'last_num_mon_in_box', 'total_healing_rew', 'died_count', 'prev_knn_rew', 'visited_pokecenter_list', 'last_10_map_ids', 'last_10_coords', 'past_events_string', 'last_10_event_ids', 'early_done', 'step_count', 'past_rewards', 'base_event_flags', 'rewarded_events_string', 'seen_map_dict', '_cut_badge', '_have_hm01', '_can_use_cut', '_surf_badge', '_have_hm03', '_can_use_surf', '_have_pokeflute', '_have_silph_scope', 'used_cut_coords_dict', '_last_item_count', '_is_box_mon_higher_level', 'hideout_elevator_maps', 'use_mart_count', 'use_pc_swap_count']
        # pyboy state file, 
        # state pkl file, 
        if not self.save_state_dir:
            return
        self.save_state_dir.mkdir(exist_ok=True)
        # state_dir naming, state_dir/{current_level}/{datetime}_{step_count}_{total_reward:5.2f}/ .state | .pkl
        if not is_failed:
            level_increment = 1
            # if self.level_completed_skip_type == 1:
            #     # special case
            #     level_increment = 2
            state_dir = self.save_state_dir / Path(f'level_{self.current_level + level_increment}')  # + 1 for next level
        else:
            # create failed folder
            state_dir = self.save_state_dir / Path(f'failed')
            state_dir.mkdir(exist_ok=True)
            state_dir = self.save_state_dir / Path(f'failed/level_{self.current_level}')
        state_dir.mkdir(exist_ok=True)
        datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        state_dir = state_dir / Path(f'{datetime_str}_{self.step_count}_{self.total_reward:5.2f}')
        state_dir.mkdir(exist_ok=True)
        # state pkl file all the required variables defined in self.reset()
        # recent_frames, agent_stats, base_explore, max_opponent_level, max_event_rew, max_level_rew, last_health, last_num_poke, last_num_mon_in_box, total_healing_rew, died_count, prev_knn_rew, visited_pokecenter_list, last_10_map_ids, last_10_coords, past_events_string, last_10_event_ids, early_done, step_count, past_rewards, base_event_flags, rewarded_events_string, seen_map_dict, _cut_badge, _have_hm01, _can_use_cut, _surf_badge, _have_hm03, _can_use_surf, _have_pokeflute, _have_silph_scope, used_cut_coords_dict, _last_item_count, _is_box_mon_higher_level, hideout_elevator_maps, use_mart_count, use_pc_swap_count
        with open(state_dir / Path('state.pkl'), 'wb') as f:
            state = {key: getattr(self, key) for key in STATES_TO_SAVE_LOAD}
            if self.enable_stage_manager:
                state['stage_manager'] = self.stage_manager
            pickle.dump(state, f)
        # pyboy state file
        with open(state_dir / Path('state.state'), 'wb') as f:
            self.pyboy.save_state(f)

    def load_state(self, state_dir):
        # STATES_TO_SAVE_LOAD
        with open(state_dir / Path('state.state'), 'rb') as f:
            self.pyboy.load_state(f)
        with open(state_dir / Path('state.pkl'), 'rb') as f:
            state = pickle.load(f)
            if 'party_level_base' not in state:
                state['party_level_base'] = 0
            if 'party_level_post' not in state:
                state['party_level_post'] = 0
            if 'secret_switch_states' not in state:
                state['secret_switch_states'] = {}
            for key in STATES_TO_SAVE_LOAD:
                # if key == 'secret_switch_states' and key not in state:
                #     self.secret_switch_states = {}
                # else:
                setattr(self, key, state[key])
            if self.enable_stage_manager:
                self.stage_manager = state['stage_manager']
        self.reset_count = 0
        # self.step_count = 0
        self.early_done = False
        self.update_last_10_map_ids()
        self.update_last_10_coords()
        self.update_seen_map_dict()
        # self.past_rewards = np.zeros(10240, dtype=np.float32)
        self.progress_reward = self.get_game_state_reward()
        self.total_reward = sum([val for _, val in self.progress_reward.items()])
        self.past_rewards[0] = self.total_reward - self.get_knn_reward_exclusion() - self.progress_reward['heal'] - self.get_dead_reward()
        # set all past reward to current total reward, so that the agent will not be penalized for the first step
        self.past_rewards[1:] = self.past_rewards[0] - (self.early_stopping_min_reward * self.reward_scale)
        self.reset_count += 1
        # if self.enable_stage_manager:
        #     self.update_stage_manager()
        
    def init_map_mem(self):
        self.seen_coords = {}
        self.perm_seen_coords = {}
        self.special_seen_coords_count = 0

    def render(self, reduce_res=True, add_memory=True, update_mem=True):
        game_pixels_render = self.screen.screen_ndarray() # (144, 160, 3)
        if reduce_res:
            game_pixels_render = game_pixels_render[:, :, 0]  # should be 3x speed up for rendering
            # game_pixels_render = (255*resize(game_pixels_render, self.output_shape)).astype(np.uint8)
            game_pixels_render = downscale_local_mean(game_pixels_render, (2, 2)).astype(np.uint8)
            if update_mem:
                reduced_frame = game_pixels_render
                self.recent_frames[0] = reduced_frame
                # collision = self.wrapper.game_area_collision() * 255
                # upscaled_collision = np.repeat(np.repeat(collision, 2, axis=0), 2, axis=1).astype(np.uint8)
                # self.recent_frames[0] = upscaled_collision
            if add_memory:
                # pad = np.zeros(
                #     shape=(self.mem_padding, self.output_shape[1], 3), 
                #     dtype=np.uint8)
                # game_pixels_render = np.concatenate(
                #     (
                #         self.create_exploration_memory(), 
                #         pad,
                #         self.create_recent_memory(),
                #         pad,
                #         rearrange(self.recent_frames, 'f h w c -> (f h) w c')
                #     ),
                #     axis=0)
                game_pixels_render = {
                    'image': self.recent_frames,
                    'minimap': self.get_minimap_obs(),
                    'minimap_sprite': self.get_minimap_sprite_obs(),
                    'minimap_warp': self.get_minimap_warp_obs(),
                    'vector': self.get_all_raw_obs(),
                    'map_ids': self.get_last_10_map_ids_obs(),
                    'map_step_since': self.get_last_10_map_step_since_obs(),
                    'item_ids': self.get_all_item_ids_obs(),
                    'item_quantity': self.get_items_quantity_obs(),
                    'poke_ids': self.get_all_pokemon_ids_obs(),
                    'poke_type_ids': self.get_all_pokemon_types_obs(),
                    'poke_move_ids': self.get_all_move_ids_obs(),
                    'poke_move_pps': self.get_all_move_pps_obs(),
                    'poke_all': self.get_all_pokemon_obs(),
                    'event_ids': self.get_all_event_ids_obs(),
                    'event_step_since': self.get_all_event_step_since_obs(),
                    # 'in_battle_mask': self.get_in_battle_mask_obs(),
                }
                # assert game_pixels_render['image'].shape == (self.frame_stacks, self.output_shape[0], self.output_shape[1]), f'game_pixels_render["image"].shape: {game_pixels_render["image"].shape}'
                # assert game_pixels_render['minimap'].shape == (14, 9, 10), f'game_pixels_render["minimap"].shape: {game_pixels_render["minimap"].shape}'
                # assert game_pixels_render['minimap_sprite'].shape == (9, 10), f'game_pixels_render["minimap_sprite"].shape: {game_pixels_render["minimap_sprite"].shape}'
                # assert game_pixels_render['minimap_warp'].shape == (9, 10), f'game_pixels_render["minimap_warp"].shape: {game_pixels_render["minimap_warp"].shape}'
                # assert game_pixels_render['vector'].shape == (58, ), f'game_pixels_render["vector"].shape: {game_pixels_render["vector"].shape}'
                # assert game_pixels_render['map_ids'].shape == (10, ), f'game_pixels_render["map_ids"].shape: {game_pixels_render["map_ids"].shape}'
                # assert game_pixels_render['map_step_since'].shape == (10, 1), f'game_pixels_render["map_step_since"].shape: {game_pixels_render["map_step_since"].shape}'
                # assert game_pixels_render['item_ids'].shape == (20, ), f'game_pixels_render["item_ids"].shape: {game_pixels_render["item_ids"].shape}'
                # assert game_pixels_render['item_quantity'].shape == (20, 1), f'game_pixels_render["item_quantity"].shape: {game_pixels_render["item_quantity"].shape}'
                # assert game_pixels_render['poke_ids'].shape == (12, ), f'game_pixels_render["poke_ids"].shape: {game_pixels_render["poke_ids"].shape}'
                # assert game_pixels_render['poke_type_ids'].shape == (12, 2), f'game_pixels_render["poke_type_ids"].shape: {game_pixels_render["poke_type_ids"].shape}'
                # assert game_pixels_render['poke_move_ids'].shape == (12, 4), f'game_pixels_render["poke_move_ids"].shape: {game_pixels_render["poke_move_ids"].shape}'
                # assert game_pixels_render['poke_move_pps'].shape == (12, 4, 2), f'game_pixels_render["poke_move_pps"].shape: {game_pixels_render["poke_move_pps"].shape}'
                # assert game_pixels_render['poke_all'].shape == (12, self.n_pokemon_features), f'game_pixels_render["poke_all"].shape: {game_pixels_render["poke_all"].shape}'
                # assert game_pixels_render['event_ids'].shape == (10, ), f'game_pixels_render["event_ids"].shape: {game_pixels_render["event_ids"].shape}'
                # assert game_pixels_render['event_step_since'].shape == (10, 1), f'game_pixels_render["event_step_since"].shape: {game_pixels_render["event_step_since"].shape}'
        return game_pixels_render
    
    @property
    def bottom_left_screen_tiles(self):
        if self._bottom_left_screen_tiles is None:
            screen_tiles = self.wrapper._get_screen_background_tilemap()
            self._bottom_left_screen_tiles = screen_tiles[1:1 + screen_tiles.shape[0]:2, ::2]-256
        return self._bottom_left_screen_tiles
    
    @property
    def bottom_right_screen_tiles(self):
        # if self._bottom_right_screen_tiles is None:
        screen_tiles = self.wrapper._get_screen_background_tilemap()
        _bottom_right_screen_tiles = screen_tiles[1:1 + screen_tiles.shape[0]:2, 1::2]-256
        return _bottom_right_screen_tiles
    
    def get_minimap_obs(self):
        if self._minimap_obs is None:
            ledges_dict = {
                'down': [54, 55],
                'left': 39,
                'right': [13, 29]
            }
            minimap = np.zeros((6, 9, 10), dtype=np.float32)
            bottom_left_screen_tiles = self.bottom_left_screen_tiles
            # walkable
            minimap[0] = self.wrapper._get_screen_walkable_matrix()
            tileset_id = self.pyboy.get_memory_value(0xd367)
            if tileset_id in [0, 3, 5, 7, 13, 14, 17, 22, 23]:  # 0 overworld, 3 forest, 
                # water
                if tileset_id == 14:  # vermilion port
                    minimap[5] = (bottom_left_screen_tiles == 20).astype(np.float32)
                else:
                    minimap[5] = np.isin(bottom_left_screen_tiles, [0x14, 0x32, 0x48]).astype(np.float32)
            
            if tileset_id == 0:  # is overworld
                # tree
                minimap[1] = (bottom_left_screen_tiles == 61).astype(np.float32)
                # ledge down
                minimap[2] = np.isin(bottom_left_screen_tiles, ledges_dict['down']).astype(np.float32)
                # ledge left
                minimap[3] = (bottom_left_screen_tiles == ledges_dict['left']).astype(np.float32)
                # ledge right
                minimap[4] = np.isin(bottom_left_screen_tiles, ledges_dict['right']).astype(np.float32)
            elif tileset_id == 7:  # is gym
                # tree
                minimap[1] = (bottom_left_screen_tiles == 80).astype(np.float32)  # 0x50
            
            # get seen_map obs
            seen_map_obs = self.get_all_seen_map_obs() # (8, 9, 10)

            minimap = np.concatenate([minimap, seen_map_obs], axis=0)  # (14, 9, 10)
            self._minimap_obs = minimap
        return self._minimap_obs
    
    @property
    def cur_seen_map(self):
        if self._cur_seen_map is None:
            cur_seen_map = np.zeros((9, 10), dtype=np.float32)
            cur_map_id = self.current_map_id - 1
            x, y = self.current_coords
            if cur_map_id not in self.seen_map_dict:
                print(f'\nERROR!!! cur_map_id: {cur_map_id} not in self.seen_map_dict')
            cur_top_left_x = x - 4
            cur_top_left_y = y - 4
            cur_bottom_right_x = x + 6
            cur_bottom_right_y = y + 5
            top_left_x = max(0, cur_top_left_x)
            top_left_y = max(0, cur_top_left_y)
            bottom_right_x = min(MAP_DICT[MAP_ID_REF[cur_map_id]]['width'], cur_bottom_right_x)
            bottom_right_y = min(MAP_DICT[MAP_ID_REF[cur_map_id]]['height'], cur_bottom_right_y)
            
            adjust_x = 0
            adjust_y = 0
            if cur_top_left_x < 0:
                adjust_x = -cur_top_left_x
            if cur_top_left_y < 0:
                adjust_y = -cur_top_left_y
            # if cur_bottom_right_x > MAP_DICT[MAP_ID_REF[cur_map_id]]['width']:
            #     adjust_x = MAP_DICT[MAP_ID_REF[cur_map_id]]['width'] - cur_bottom_right_x
            # if cur_bottom_right_y > MAP_DICT[MAP_ID_REF[cur_map_id]]['height']:
            #     adjust_y = MAP_DICT[MAP_ID_REF[cur_map_id]]['height'] - cur_bottom_right_y

            cur_seen_map[adjust_y:adjust_y + bottom_right_y - top_left_y, adjust_x:adjust_x + bottom_right_x - top_left_x] = self.seen_map_dict[cur_map_id][top_left_y:bottom_right_y, top_left_x:bottom_right_x]
            self._cur_seen_map = cur_seen_map
        return self._cur_seen_map
    
    def get_seen_map_obs(self, steps_since=-1):
        cur_seen_map = self.cur_seen_map.copy()

        last_step_count = self.step_count - 1
        if steps_since == -1:  # set all seen tiles to 1
            cur_seen_map[cur_seen_map > 0] = 1
        else:
            if steps_since > last_step_count:
                cur_seen_map[cur_seen_map > 0] = (cur_seen_map[cur_seen_map > 0] + (steps_since - last_step_count)) / steps_since
            else:
                cur_seen_map = (cur_seen_map - (last_step_count - steps_since)) / steps_since
                cur_seen_map[cur_seen_map < 0] = 0
        return np.expand_dims(cur_seen_map, axis=0)
    
    def get_all_seen_map_obs(self):
        if self.is_warping:
            return np.zeros((8, 9, 10), dtype=np.float32)
        
        # workaround for seen map xy axis bug
        cur_map_id = self.current_map_id - 1
        x, y = self.current_coords
        if y >= self.seen_map_dict[cur_map_id].shape[0] or x >= self.seen_map_dict[cur_map_id].shape[1]:
            # print(f'ERROR1z: x: {x}, y: {y}, cur_map_id: {cur_map_id} ({MAP_ID_REF[cur_map_id]}), seen_map_dict[cur_map_id].shape: {self.seen_map_dict[cur_map_id].shape}')
            # print(f'ERROR2z: last 10 map ids: {self.last_10_map_ids}')
            return np.zeros((8, 9, 10), dtype=np.float32)

        map_10 = self.get_seen_map_obs(steps_since=10)  # (1, 9, 10)
        map_50 = self.get_seen_map_obs(steps_since=50)  # (1, 9, 10)
        map_500 = self.get_seen_map_obs(steps_since=500)  # (1, 9, 10)
        map_5_000 = self.get_seen_map_obs(steps_since=5_000)  # (1, 9, 10)
        map_50_000 = self.get_seen_map_obs(steps_since=50_000)  # (1, 9, 10)
        map_500_000 = self.get_seen_map_obs(steps_since=500_000)  # (1, 9, 10)
        map_5_000_000 = self.get_seen_map_obs(steps_since=5_000_000)  # (1, 9, 10)
        map_50_000_000 = self.get_seen_map_obs(steps_since=50_000_000)  # (1, 9, 10)
        return np.concatenate([map_10, map_50, map_500, map_5_000, map_50_000, map_500_000, map_5_000_000, map_50_000_000], axis=0) # (8, 9, 10)
    
    def assign_new_sprite_in_sprite_minimap(self, minimap, sprite_id, x, y):
        x, y = self.current_coords
        top_left_x = x - 4
        top_left_y = y - 4
        if x >= top_left_x and x < top_left_x + 10 and y >= top_left_y and y < top_left_y + 9:
            minimap[y - top_left_y, x - top_left_x] = sprite_id
    
    @property
    def minimap_sprite(self):
        if self._minimap_sprite is None:
            minimap_sprite = np.zeros((9, 10), dtype=np.int16)
            sprites = self.wrapper._sprites_on_screen()
            for idx, s in enumerate(sprites):
                if (idx + 1) % 4 != 0:
                    continue
                minimap_sprite[s.y // 16, s.x // 16] = (s.tiles[0].tile_identifier + 1) / 4
            map_id = self.current_map_id - 1
            # special case for vermilion gym
            if map_id == 0x5C and not self.read_bit(0xD773, 0):
                trashcans_coords = [
                    (1, 7), (1, 9), (1, 11), 
                    (3, 7), (3, 9), (3, 11),
                    (5, 7), (5, 9), (5, 11),
                    (7, 7), (7, 9), (7, 11),
                    (9, 7), (9, 9), (9, 11),
                ]
                first_can = self.read_ram_m(RAM.wFirstLockTrashCanIndex)
                if self.read_bit(0xD773, 1):
                    second_can = self.read_ram_m(RAM.wSecondLockTrashCanIndex)
                    first_can_coords = trashcans_coords[second_can]
                else:
                    first_can_coords = trashcans_coords[first_can]
                self.assign_new_sprite_in_sprite_minimap(minimap_sprite, 384, first_can_coords[0], first_can_coords[1])
            # special case for pokemon mansion secret switch
            elif map_id == 0xA5:
                # 1F, secret switch id 383
                # secret switch 1: 2, 5
                self.assign_new_sprite_in_sprite_minimap(minimap_sprite, 383, 2, 5)
            elif map_id == 0xD6:
                # 2F, secret switch id 383
                # secret switch 1: 2, 11
                self.assign_new_sprite_in_sprite_minimap(minimap_sprite, 383, 2, 11)
            elif map_id == 0xD7:
                # 3F, secret switch id 383
                # secret switch 1: 10, 5
                self.assign_new_sprite_in_sprite_minimap(minimap_sprite, 383, 10, 5)
            elif map_id == 0xD8:
                # B1F, secret switch id 383
                # secret switch 1: 20, 3
                # secret switch 2: 18, 25
                self.assign_new_sprite_in_sprite_minimap(minimap_sprite, 383, 20, 3)
                self.assign_new_sprite_in_sprite_minimap(minimap_sprite, 383, 18, 25)
            self._minimap_sprite = minimap_sprite
        return self._minimap_sprite
    
    def get_minimap_sprite_obs(self):
        # minimap_sprite = np.zeros((9, 10), dtype=np.int16)
        # sprites = self.wrapper._sprites_on_screen()
        # for idx, s in enumerate(sprites):
        #     if (idx + 1) % 4 != 0:
        #         continue
        #     minimap_sprite[s.y // 16, s.x // 16] = (s.tiles[0].tile_identifier + 1) / 4
        # return minimap_sprite
        return self.minimap_sprite
    
    def get_minimap_warp_obs(self):
        if self._minimap_warp_obs is None:
            minimap_warp = np.zeros((9, 10), dtype=np.int16)
            # self.current_map_id
            cur_map_id = self.current_map_id - 1
            map_name = MAP_ID_REF[cur_map_id]
            if cur_map_id == 255:
                print(f'hard stuck map_id 255, force ES')
                self.early_done = True
                return minimap_warp
            # if map_name not in WARP_DICT:
            #     print(f'ERROR: map_name: {map_name} not in MAP_DICT, last 10 map ids: {self.last_10_map_ids}')
            #     # self.save_all_states(is_failed=True)
            #     # raise ValueError(f'map_name: {map_name} not in MAP_DICT, last 10 map ids: {self.last_10_map_ids}')
            #     return minimap_warp
            warps = WARP_DICT[map_name]
            if not warps:
                return minimap_warp
            x, y = self.current_coords
            top_left_x = max(0, x - 4)
            top_left_y = max(0, y - 4)
            bottom_right_x = min(MAP_DICT[MAP_ID_REF[cur_map_id]]['width'], x + 5)
            bottom_right_y = min(MAP_DICT[MAP_ID_REF[cur_map_id]]['height'], y + 4)
            # patched warps, read again from ram
            if cur_map_id in [0xCB, 0xEC]:  # ROCKET_HIDEOUT_ELEVATOR 0xCB, SIPLH_CO_ELEVATOR 0xEC
                warps = []
                n_warps = self.read_ram_m(RAM.wNumberOfWarps)  # wNumberOfWarps
                for i in range(n_warps):
                    warp_addr = RAM.wWarpEntries.value + i * 4
                    warp_y = self.read_m(warp_addr + 0)
                    warp_x = self.read_m(warp_addr + 1)
                    warp_warp_id = self.read_m(warp_addr + 2)
                    warp_map_id = self.read_m(warp_addr + 3)
                    if warp_map_id in [199, 200, 201, 202] and warp_map_id not in self.hideout_elevator_maps:
                        self.hideout_elevator_maps.append(warp_map_id)
                    warps.append({
                        'x': warp_x,
                        'y': warp_y,
                        'warp_id': warp_warp_id,
                        'target_map_name': MAP_ID_REF[warp_map_id],
                    })

            for warp in warps:
                if warp['x'] >= top_left_x and warp['x'] <= bottom_right_x and warp['y'] >= top_left_y and warp['y'] <= bottom_right_y:
                    if warp['target_map_name'] != 'LAST_MAP':
                        target_map_name = warp['target_map_name']
                    else:
                        last_map_id = self.read_m(0xd365)  # wLastMap
                        target_map_name = MAP_ID_REF[last_map_id]
                    warp_id = warp['warp_id'] - 1
                    warp_name = f'{target_map_name}@{warp_id}'
                    if warp_name in WARP_ID_DICT:
                        actual_warp_id = WARP_ID_DICT[warp_name] + 1  # 0 is reserved for no warp / padding
                    else:
                        actual_warp_id = 829
                        # if warp_name not in ['ROUTE_22@1']:  # ignore expected bugged warps, workaround-ed
                        if warp_name in ['SAFFRON_CITY@9']:  # ignore expected bugged warps, workaround-ed
                            actual_warp_id = 828
                        else:
                            print(f'warp_name: {warp_name} not in WARP_ID_DICT')
                    minimap_warp[warp['y'] - top_left_y, warp['x'] - top_left_x] = actual_warp_id
            self._minimap_warp_obs = minimap_warp
        return self._minimap_warp_obs
    
    @property
    def is_warping(self):
        if self._is_warping is None:
            hdst_map = self.read_m(0xFF8B)
            if self.read_ram_bit(RAM.wd736, 2) == 1:
                self._is_warping = hdst_map == 255 or self.read_ram_m(RAM.wCurMap) == hdst_map
            elif self.read_ram_m(RAM.wStandingOnWarpPadOrHole) == 1:
                self._is_warping = True
            else:
                x, y = self.current_coords
                n_warps = self.read_m(0xd3ae)  # wNumberOfWarps
                for i in range(n_warps):
                    warp_addr = RAM.wWarpEntries.value + i * 4
                    if self.read_m(warp_addr + 0) == y and self.read_m(warp_addr + 1) == x:
                        self._is_warping = hdst_map == 255 or self.read_ram_m(RAM.wCurMap) == hdst_map
                        break
            # self._is_warping = self.read_bit(0xd736, 2) == 1 and self.read_m(0xFF8B) == self.read_m(0xD35E)
        return self._is_warping
    
    def update_seen_map_dict(self):
        # if self.get_minimap_warp_obs()[4, 4] != 0:
        #     return
        cur_map_id = self.current_map_id - 1
        x, y = self.current_coords
        if cur_map_id not in self.seen_map_dict:
            self.seen_map_dict[cur_map_id] = np.zeros((MAP_DICT[MAP_ID_REF[cur_map_id]]['height'], MAP_DICT[MAP_ID_REF[cur_map_id]]['width']), dtype=np.float32)
            
        # # do not update if is warping
        if not self.is_warping:
            if y >= self.seen_map_dict[cur_map_id].shape[0] or x >= self.seen_map_dict[cur_map_id].shape[1]:
                self.stuck_cnt += 1
                print(f'ERROR1: x: {x}, y: {y}, cur_map_id: {cur_map_id} ({MAP_ID_REF[cur_map_id]}), map.shape: {self.seen_map_dict[cur_map_id].shape}')
                if self.stuck_cnt > 50:
                    print(f'stucked for > 50 steps, force ES')
                    self.early_done = True
                    self.stuck_cnt = 0
                # print(f'ERROR2: last 10 map ids: {self.last_10_map_ids}')
            else:
                self.stuck_cnt = 0
                self.seen_map_dict[cur_map_id][y, x] = self.step_count
    
    def update_last_10_map_ids(self):
        current_modified_map_id = self.read_m(0xD35E) + 1
        # check if current_modified_map_id is in last_10_map_ids
        if current_modified_map_id == self.last_10_map_ids[0][0]:
            return
        else:
            # if self.last_10_map_ids[0][0] != 0:
            #     print(f'map changed from {MAP_ID_REF[self.last_10_map_ids[0][0] - 1]} to {MAP_ID_REF[current_modified_map_id - 1]} at step {self.step_count}')
            self.last_10_map_ids = np.roll(self.last_10_map_ids, 1, axis=0)
            self.last_10_map_ids[0] = [current_modified_map_id, self.step_count]
            map_id = current_modified_map_id - 1
            if map_id in [0x6C, 0xC2, 0xC6, 0x22]:
                self.minor_patch_victory_road()
            # elif map_id == 0x09:
            if map_id not in [0xF5, 0xF6, 0xF7, 0x71, 0x78]:
                if self.last_10_map_ids[1][0] - 1 in [0xF5, 0xF6, 0xF7, 0x71, 0x78]:
                    # lost in elite 4
                    self.elite_4_lost = True
                    self.elite_4_started_step = None
            if map_id == 0xF5:
                # elite four first room
                # reset elite 4 lost flag
                if self.elite_4_lost:
                    self.elite_4_lost = False
                if self.elite_4_started_step is None:
                    self.elite_4_started_step = self.step_count
        
    def minor_patch_victory_road(self):
        address_bits = [
            # victory road
            [0xD7EE, 0],
            [0xD7EE, 7],
            [0xD813, 0],
            [0xD813, 6],
            [0xD869, 7],
        ]
        for ab in address_bits:
            event_value = self.read_m(ab[0])
            self.pyboy.set_memory_value(ab[0], self.set_bit(event_value, ab[1]))

    def update_last_10_coords(self):
        current_coord = np.array([self.read_m(0xD362), self.read_m(0xD361)])
        # check if current_coord is in last_10_coords
        if (current_coord == self.last_10_coords[0]).all():
            return
        else:
            self.last_10_coords = np.roll(self.last_10_coords, 1, axis=0)
            self.last_10_coords[0] = current_coord

    def update_cut_badge(self):
        if not self._cut_badge:
            self._cut_badge = self.read_ram_bit(RAM.wObtainedBadges, 1) == 1

    def update_surf_badge(self):
        if not self._cut_badge:
            return
        if not self._surf_badge:
            self._surf_badge = self.read_ram_bit(RAM.wObtainedBadges, 4) == 1

    def is_battle_actionable(self) -> Union[bool, str]:
        tile_map_base = 0xc3a0
        text_box_id = self.read_ram_m(RAM.wTextBoxID)
        is_safari_battle = self.read_ram_m(RAM.wBattleType) == 2
        if is_safari_battle:
            if text_box_id == 0x1b and \
                self.read_m(tile_map_base + 14 * 20 + 14) == CHARMAP["B"] and \
                self.read_m(tile_map_base + 14 * 20 + 15) == CHARMAP["A"]:
                return True
            elif text_box_id == 0x14 and \
                self.read_ram_m(RAM.wTopMenuItemX) == 15 and \
                self.read_ram_m(RAM.wTopMenuItemY) == 8 and \
                self.read_m(tile_map_base + 14 * 20 + 8) == CHARMAP["n"] and \
                self.read_m(tile_map_base + 14 * 20 + 9) == CHARMAP["i"] and \
                self.read_m(tile_map_base + 14 * 20 + 10) == CHARMAP["c"]:
                # nickname for caught pokemon
                return 'NICKNAME'
        elif text_box_id == 0x0b and \
            self.read_m(tile_map_base + 14 * 20 + 16) == CHARMAP["<PK>"] and \
            self.read_m(tile_map_base + 14 * 20 + 17) == CHARMAP["<MN>"]:
            # battle menu
            # if self.print_debug: print(f'is in battle menu at step {self.step_count}')
            return True
        elif text_box_id in [0x0b, 0x01] and \
            self.read_m(tile_map_base + 17 * 20 + 4) == CHARMAP["└"] and \
            self.read_m(tile_map_base + 8 * 20 + 10) == CHARMAP["┐"] and \
            self.read_ram_m(RAM.wTopMenuItemX) == 5 and \
            self.read_ram_m(RAM.wTopMenuItemY) == 12:
            # fight submenu
            # if self.print_debug: print(f'is in fight submenu at step {self.step_count}')
            return True
        elif text_box_id == 0x0d and \
            self.read_m(tile_map_base + 2 * 20 + 4) == CHARMAP["┌"] and \
            self.read_ram_m(RAM.wTopMenuItemX) == 5 and \
            self.read_ram_m(RAM.wTopMenuItemY) == 4:
            # bag submenu
            # if self.print_debug: print(f'is in bag submenu at step {self.step_count}')
            return True
        elif text_box_id == 0x01 and \
            self.read_m(tile_map_base + 14 * 20 + 1) == CHARMAP["C"] and \
            self.read_m(tile_map_base + 14 * 20 + 2) == CHARMAP["h"] and \
            self.read_m(tile_map_base + 14 * 20 + 3) == CHARMAP["o"]:
            # choose pokemon
            # if self.print_debug: print(f'is in choose pokemon at step {self.step_count}')
            return True
        elif text_box_id == 0x01 and \
            self.read_m(tile_map_base + 14 * 20 + 1) == CHARMAP["B"] and \
            self.read_m(tile_map_base + 14 * 20 + 2) == CHARMAP["r"] and \
            self.read_m(tile_map_base + 14 * 20 + 3) == CHARMAP["i"]:
            # choose pokemon after opponent fainted
            # choose pokemon after party pokemon fainted
            # if self.print_debug: print(f'is in choose pokemon after opponent fainted at step {self.step_count}')
            return True
        elif text_box_id == 0x01 and \
            self.read_m(tile_map_base + 14 * 20 + 1) == CHARMAP["U"] and \
            self.read_m(tile_map_base + 14 * 20 + 2) == CHARMAP["s"] and \
            self.read_m(tile_map_base + 14 * 20 + 3) == CHARMAP["e"] and \
            self.read_m(tile_map_base + 16 * 20 + 8) == CHARMAP["?"]:
            # use item in party submenu
            # if self.print_debug: print(f'is in use item in party submenu at step {self.step_count}')
            return True
        elif text_box_id == 0x0c and \
            self.read_m(tile_map_base + 12 * 20 + 13) == CHARMAP["S"] and \
            self.read_m(tile_map_base + 12 * 20 + 14) == CHARMAP["W"]:
            # switch pokemon
            return 'SWITCH'
        elif text_box_id == 0x14 and \
            self.read_ram_m(RAM.wTopMenuItemX) == 1 and \
            self.read_ram_m(RAM.wTopMenuItemY) == 8 and \
            self.read_m(tile_map_base + 16 * 20 + 1) == CHARMAP["c"] and \
            self.read_m(tile_map_base + 16 * 20 + 2) == CHARMAP["h"] and \
            self.read_m(tile_map_base + 16 * 20 + 15) == CHARMAP["?"]:
            # change pokemon yes no menu
            # if self.print_debug: print(f'is in change pokemon yes no menu at step {self.step_count}')
            return True
        elif text_box_id == 0x14 and \
            self.read_ram_m(RAM.wTopMenuItemX) == 15 and \
            self.read_ram_m(RAM.wTopMenuItemY) == 8 and \
            self.read_m(tile_map_base + 14 * 20 + 9) == CHARMAP["m"] and \
            self.read_m(tile_map_base + 14 * 20 + 10) == CHARMAP["a"] and \
            self.read_m(tile_map_base + 14 * 20 + 11) == CHARMAP["k"]:
            # make room for new move
            return 'NEW_MOVE'
        elif text_box_id == 0x01 and \
            self.read_ram_m(RAM.wTopMenuItemX) == 5 and \
            self.read_ram_m(RAM.wTopMenuItemY) == 8 and \
            self.read_m(tile_map_base + 16 * 20 + 10) == CHARMAP["t"] and \
            self.read_m(tile_map_base + 16 * 20 + 11) == CHARMAP["e"] and \
            self.read_m(tile_map_base + 16 * 20 + 12) == CHARMAP["n"] and \
            self.read_m(tile_map_base + 16 * 20 + 13) == CHARMAP["?"] and \
            self.read_ram_m(RAM.wMaxMenuItem) == 3:
            # choose move to replace
            return 'REPLACE_MOVE'
        elif text_box_id == 0x14 and \
            self.read_ram_m(RAM.wTopMenuItemX) == 15 and \
            self.read_ram_m(RAM.wTopMenuItemY) == 8 and \
            self.read_m(tile_map_base + 14 * 20 + 1) == CHARMAP["A"] and \
            self.read_m(tile_map_base + 14 * 20 + 2) == CHARMAP["b"] and \
            self.read_m(tile_map_base + 14 * 20 + 3) == CHARMAP["a"]:
            # do not learn move
            return 'ABANDON_MOVE'
        elif text_box_id == 0x14 and \
            self.read_ram_m(RAM.wTopMenuItemX) == 15 and \
            self.read_ram_m(RAM.wTopMenuItemY) == 8 and \
            self.read_m(tile_map_base + 14 * 20 + 8) == CHARMAP["n"] and \
            self.read_m(tile_map_base + 14 * 20 + 9) == CHARMAP["i"] and \
            self.read_m(tile_map_base + 14 * 20 + 10) == CHARMAP["c"]:
            # nickname for caught pokemon
            return 'NICKNAME'
        return False
    
    def step(self, action):
        if action == self.swap_button_index:
            self.scripted_roll_party()
        else:
            # if self.auto_skip_anim:
            #     self.run_action_on_emulator(action)
            is_action_taken = False
            if self.is_in_battle():
                tile_map_base = 0xc3a0
                actionable_cnt = 0
                self.print_debug = False
                while True:
                    is_actionable = self.is_battle_actionable()
                    actionable_cnt += 1
                    # if actionable_cnt > 120:  # so far the longest non-actionable loop is around 90
                    #     self.print_debug = True
                    #     self.save_screenshot(f'{str(is_actionable)}_actionable_debug')
                    #     print(f'ERROR: actionable_cnt > 120 at step {self.step_count}')
                    #     if actionable_cnt > 200:
                    #         break
                    if not self.is_in_battle():
                        # print(f'battle ended at step {self.step_count}')
                        break
                    elif self.read_m(0xFFB0) != 0 and \
                        self.read_m(tile_map_base + 12 * 20 + 0) != CHARMAP["┌"]:
                        # not in any menu
                        # likely battle ended
                        # print(f'not in any menu at step {self.step_count}')
                        break
                    elif is_actionable is True:
                        # print(f'is_actionable is True at step {self.step_count}')
                        break
                    elif is_actionable is False:
                        # if self.auto_skip_anim:
                        self.run_action_on_emulator(4)
                        is_action_taken = True
                        # else:
                        #     break
                    elif is_actionable == 'SWITCH':
                        # auto switch
                        if self.read_ram_m(RAM.wCurrentMenuItem) != 0:
                            self.pyboy.set_memory_value(RAM.wCurrentMenuItem.value, 0)
                        self.run_action_on_emulator(4)
                        is_action_taken = True
                    elif is_actionable == 'NEW_MOVE':
                        # auto make room for new move
                        self.run_action_on_emulator(4)
                        is_action_taken = True
                    elif is_actionable == 'REPLACE_MOVE':
                        # wTopMenuItemX: 5
                        # wTopMenuItemY: 8
                        # TextBoxID: 01
                        # wMaxMenuItem 3
                        # wMoves 4 moves id
                        # wMoveNum learning move id
                        # wCurrentMenuItem selected move index to replace
                        moves = [self.read_m(RAM.wMoves.value + i) for i in range(4)]
                        moves_power = []
                        party_pos = self.read_ram_m(RAM.wWhichPokemon)
                        party_type_addr = 0xD170
                        # 2 types per pokemon
                        ptypes = [self.read_m(party_type_addr + (party_pos * 44) + i) for i in range(2)]
                        for move_id in moves:
                            if move_id == 0:
                                moves_power.append(0)
                            elif move_id not in MOVES_INFO_DICT:
                                moves_power.append(0)
                                print(f'\nERROR: move_id: {move_id} not in MOVES_INFO_DICT')
                            else:
                                this_move = MOVES_INFO_DICT[move_id]
                                this_move_power = this_move['power']
                                if this_move['type_id'] in ptypes and this_move['raw_power'] > 0:
                                    this_move_power *= 1.5
                                moves_power.append(this_move_power)
                        new_move_id = self.read_ram_m(RAM.wMoveNum)
                        new_move = MOVES_INFO_DICT[new_move_id]
                        new_move_power = new_move['power']
                        if new_move['type_id'] in ptypes and new_move['raw_power'] > 0:
                            new_move_power *= 1.5
                        if new_move_power > min(moves_power):
                            # replace the move with the lowest power
                            min_power = min(moves_power)
                            min_power_idx = moves_power.index(min_power)
                            self.pyboy.set_memory_value(RAM.wCurrentMenuItem.value, min_power_idx)
                            self.run_action_on_emulator(4)
                            is_action_taken = True
                        else:
                            # do not replace, press B
                            self.run_action_on_emulator(5)
                            is_action_taken = True
                    elif is_actionable == 'ABANDON_MOVE':
                        # auto abandon move
                        self.run_action_on_emulator(4)
                        is_action_taken = True
                    elif is_actionable == 'NICKNAME':
                        # auto decline nickname
                        self.run_action_on_emulator(5)
                        is_action_taken = True
                    else:
                        print(f'ERROR: unknown is_actionable: {is_actionable}')
                        self.save_screenshot(f'unknown_is_actionable_{str(is_actionable)}')
                        break
                    self.update_heal_reward()
                    self.last_health = self.read_hp_fraction()
                    if not self.auto_skip_anim:
                        break
                    elif actionable_cnt >=  self.auto_skip_anim_frames:
                        # auto_skip_anim enabled
                        break
            # elif self.battle_type == 2:
            #     # safari battle
            #     pass
            else:
                if self.can_auto_press_a():
                    self.run_action_on_emulator(4)
                    is_action_taken = True
            if not is_action_taken:  # not self.auto_skip_anim and 
                self.run_action_on_emulator(action)
        self.init_caches()
        self.check_if_early_done()
        self.check_if_level_completed()

        # self.append_agent_stats(action)

        self.update_cut_badge()
        self.update_surf_badge()
        self.update_last_10_map_ids()
        self.update_last_10_coords()
        self.update_seen_map_dict()
        self.update_visited_pokecenter_list()
        self.recent_frames = np.roll(self.recent_frames, 1, axis=0)
        self.minor_patch()
        if self.enable_item_manager:
            self.scripted_manage_items()
        obs_memory = self.render()


        # if self.use_screen_explore:
        #     # trim off memory from frame for knn index
        #     obs_flat = obs_memory['image'].flatten().astype(np.float32)
        #     self.update_frame_knn_index(obs_flat)
        # else:
        self.update_seen_coords()
            
        self.update_heal_reward()
        self.update_num_poke()
        self.update_num_mon_in_box()
        if self.enable_stage_manager:
            self.update_stage_manager()

        new_reward = self.update_reward()
        
        self.last_health = self.read_hp_fraction()

        # shift over short term reward memory
        # self.recent_memory = np.roll(self.recent_memory, 3)
        # self.recent_memory[0, 0] = min(new_prog[0] * 64, 255)
        # self.recent_memory[0, 1] = min(new_prog[1] * 64, 255)
        # self.recent_memory[0, 2] = min(new_prog[2] * 128, 255)

        # self.update_past_events()  # done in update_reward's update_max_event_rew

        self.past_events_string = self.all_events_string

        # record past rewards
        self.past_rewards = np.roll(self.past_rewards, 1)
        self.past_rewards[0] = self.total_reward - self.get_knn_reward_exclusion() - self.progress_reward['heal'] - self.get_dead_reward()

        step_limit_reached = self.check_if_done()

        # if step_limit_reached:
        #     self._last_episode_stats = self.get_stats()
        
        # if not self.warmed_up and self.randomize_first_ep_split_cnt and \
        #     self.step_count and self.step_count % (self.max_steps // self.randomize_first_ep_split_cnt) == 0 and \
        #     1.0 / self.randomize_first_ep_split_cnt > np.random.rand():
        #     # not warmed up yet
        #     # check if step count reached the checkpoint of randomize_first_ep_split_cnt
        #     # if reached, randomly decide to end the episode based on randomize_first_ep_split_cnt
        #     step_limit_reached = True
        #     self.warmed_up = True
        #     print(f'randomly end episode at step {self.step_count} with randomize_first_ep_split_cnt: {self.randomize_first_ep_split_cnt}')
        if not self.warmed_up and self.randomize_first_ep_split_cnt and \
            self.step_count and self.step_count % ((self.max_steps // self.randomize_first_ep_split_cnt) * (self.env_id + 1)) == 0:
            # not warmed up yet
            # check if step count reached the checkpoint of randomize_first_ep_split_cnt
            # if reached, randomly decide to end the episode based on randomize_first_ep_split_cnt
            step_limit_reached = True
            self.warmed_up = True
            print(f'randomly end episode at step {self.step_count} with randomize_first_ep_split_cnt: {self.randomize_first_ep_split_cnt}')

        if self.level_completed:
            if not self.level_manager_eval_mode or self.current_level == 7:
                step_limit_reached = True
                print(f'BEATEN CHAMPIOM at step {self.env_id}:{self.step_count}')
            print(f'\nlevel {self.current_level} completed at step {self.step_count}')

        self.save_and_print_info(step_limit_reached, obs_memory)

        if self.level_completed and self.level_manager_eval_mode:
            self.current_level += 1

        self.step_count += 1

        if not self.level_manager_initialized:
            self.level_manager_initialized = True

        return obs_memory, new_reward*0.1, False, step_limit_reached, {}
    
    def init_caches(self):
        # for cached properties
        self._all_events_string = ''
        self._battle_type = None
        self._cur_seen_map = None
        self._minimap_warp_obs = None
        self._is_warping = None
        self._items_in_bag = None
        self._minimap_obs = None
        self._minimap_sprite = None
        self._bottom_left_screen_tiles = None
        self._num_mon_in_box = None

    def check_if_level_completed(self):
        self.level_completed = False
        self.level_completed = self.scripted_level_manager()

    @property
    def num_mon_in_box(self):
        if self._num_mon_in_box is None:
            self._num_mon_in_box = self.read_m(0xda80)
        return self._num_mon_in_box
    
    def get_first_diff_index(self, arr1, arr2):
        for i in range(len(arr1)):
            if arr1[i] != arr2[i] and arr2[i] == '1':
                return i
        return -1
    
    # def update_past_events(self):
    #     if self.past_events_string and self.past_events_string != self.all_events_string:
    #         first_diff_index = self.get_first_diff_index(self.past_events_string, self.all_events_string)
    #         assert len(self.all_events_string) == len(self.past_events_string), f'len(self.all_events_string): {len(self.all_events_string)}, len(self.past_events_string): {len(self.past_events_string)}'
    #         if first_diff_index != -1:
    #             self.last_10_event_ids = np.roll(self.last_10_event_ids, 1, axis=0)
    #             self.last_10_event_ids[0] = [first_diff_index, self.step_count]
    #             print(f'new event at step {self.step_count}, event: {self.last_10_event_ids[0]}')
    
    def is_in_start_menu(self) -> bool:
        menu_check_dict = {
            'hWY': self.read_m(0xFFB0) == 0,
            'wFontLoaded': self.read_m(0xCFC4) == 1,
            'wUpdateSpritesEnabled': self.read_m(0xcfcb) == 1,
            'wMenuWatchedKeys': self.read_m(0xcc29) == 203,
            'wTopMenuItemY': self.read_m(0xcc24) == 2,
            'wTopMenuItemX': self.read_m(0xcc25) == 11,
        }
        for val in menu_check_dict.values():
            if not val:
                return False
        return True
        # return self.read_m(0xD057) == 0x0A

    def get_menu_restricted_action(self, action: int) -> int:
        if not self.is_in_battle():
            if self.is_in_start_menu():
                # not in battle and in start menu
                # if wCurrentMenuItem == 1, then up / down will be changed to down
                # if wCurrentMenuItem == 2, then up / down will be changed to up
                current_menu_item = self.read_m(0xCC26)
                if current_menu_item not in [1, 2]:
                    print(f'\nWarning! current start menu item: {current_menu_item}, not 1 or 2')
                    # self.save_screenshot('start_menu_item_not_1_or_2')
                    # do nothing, return action
                    return action
                if action < 4:
                    # any arrow key will be changed to down if wCurrentMenuItem == 1
                    # any arrow key will be changed to up if wCurrentMenuItem == 2
                    if current_menu_item == 1:
                        action = 0  # down
                    elif current_menu_item == 2:
                        action = 3  # up
            elif action == 6:
                # not in battle and start menu, pressing START
                # opening menu, always set to 1
                self.pyboy.set_memory_value(0xCC2D, 1)  # wBattleAndStartSavedMenuItem
        return action
    
    @property
    def can_use_cut(self):
        # return self.read_m(0xD2E2) == 1
        # check badge, store last badge count, if changed, check if can use cut, bit 1, save permanently
        if not self._can_use_cut:
            if self._cut_badge:
                if not self._have_hm01:
                    self._have_hm01 = 0xc4 in self.get_items_in_bag()
                if self._have_hm01:
                    self._can_use_cut = True
            # self._can_use_cut = self._cut_badge is True and 0xc4 in self.get_items_in_bag()
        return self._can_use_cut
    
    @property
    def can_use_surf(self):
        if not self._can_use_surf:
            if self._surf_badge:
                if not self._have_hm03:
                    self._have_hm03 = 0xC6 in self.get_items_in_bag()
                if self._have_hm03:
                    self._can_use_surf = True
        return self._can_use_surf
    
    @property
    def have_silph_scope(self):
        if self.can_use_cut and not self._have_silph_scope:
            self._have_silph_scope = 0x48 in self.get_items_in_bag()
        return self._have_silph_scope
    
    @property
    def can_use_flute(self):
        if self.can_use_cut and not self._have_pokeflute:
            self._have_pokeflute = 0x49 in self.get_items_in_bag()
        return self._have_pokeflute

    def scripted_routine_cut(self, action):
        if not self.can_use_cut:
            return
        # TURN THIS ON OR ELSE IT WILL AUTO CUT
        if action != 4:
            return
        
        if self.read_m(0xFFB0) == 0:
            # in menu
            return

        # can only be used in overworld and gym
        tile_id = self.read_ram_m(RAM.wCurMapTileset)
        use_cut_now = False
        cut_tile = -1
        
        # check if wTileInFrontOfPlayer is tree, 0x3d in overworld, 0x50 in gym
        if tile_id in [0, 7]:  # overworld, gym
            minimap_tree = self.get_minimap_obs()[1]
            facing_direction = self.read_m(0xC109)  # wSpritePlayerStateData1FacingDirection
            if facing_direction == 0:  # down
                tile_infront = minimap_tree[5, 4]
            elif facing_direction == 4:  # up
                tile_infront = minimap_tree[3, 4]
            elif facing_direction == 8:  # left
                tile_infront = minimap_tree[4, 3]
            elif facing_direction == 12:  # right
                tile_infront = minimap_tree[4, 5]
            # tile_infront = self.read_ram_m(RAM.wTileInFrontOfPlayer)
            if tile_id == 0 and tile_infront == 1:
                use_cut_now = True
                cut_tile = 0x3d
            elif tile_id == 7 and tile_infront == 1:
                use_cut_now = True
                cut_tile = 0x50
        if use_cut_now:
            self.pyboy.set_memory_value(RAM.wBattleAndStartSavedMenuItem.value, 1)  # set to Pokemon
            self.run_action_on_emulator(action=10, emulated=WindowEvent.PRESS_BUTTON_START)
            self.run_action_on_emulator(action=4, emulated=WindowEvent.PRESS_BUTTON_A)
            for _ in range(3):
                self.pyboy.set_memory_value(RAM.wFieldMoves.value, 1)  # set first field move to cut
                self.pyboy.set_memory_value(RAM.wWhichPokemon.value, 0)  # first pokemon
                self.pyboy.set_memory_value(RAM.wMaxMenuItem.value, 3)  # max menu item
                self.run_action_on_emulator(action=4, emulated=WindowEvent.PRESS_BUTTON_A)
            # post check if wActionResultOrTookBattleTurn == 1
            if self.read_ram_m(RAM.wActionResultOrTookBattleTurn) == 1 and self.read_ram_m(RAM.wCutTile) == cut_tile:
                self.used_cut_coords_dict[f'x:{self.current_coords[0]} y:{self.current_coords[1]} m:{self.current_map_id}'] = self.step_count
                # print(f'\ncut used at step {self.step_count}, coords: {self.current_coords}, map: {MAP_ID_REF[self.current_map_id - 1]}, used_cut_coords_dict: {self.used_cut_coords_dict}')
            else:
                pass
                # print(f'\nERROR! cut failed, actioresult: {self.read_ram_m(RAM.wActionResultOrTookBattleTurn)}, wCutTile: {self.read_ram_m(RAM.wCutTile)}, xy: {self.current_coords}, map: {MAP_ID_REF[self.current_map_id - 1]}')
    
    def scripted_routine_surf(self, action):
        if not self.can_use_surf:
            return
        # TURN THIS ON OR ELSE IT WILL AUTO SURF
        if action != 4:
            return
        
        if self.read_m(0xFFB0) == 0 or self.read_ram_m(RAM.wWalkBikeSurfState) == 2:
            # in menu
            # or already surfing
            return

        # can only be used in overworld and gym
        tile_id = self.read_ram_m(RAM.wCurMapTileset)
        use_surf_now = False

        if tile_id not in [0, 3, 5, 7, 13, 14, 17, 22, 23]:
            return

        # surf_tile = -1
        # TilePairCollisionsWater
        # db FOREST, $14, $2E
        # db FOREST, $48, $2E
        # db CAVERN, $14, $05
        facing_direction = self.read_m(0xC109)  # wSpritePlayerStateData1FacingDirection
        # if tile_id in [0, 3, 5, 7, 13, 14, 17, 22, 23]:
        # minimap_water = self.get_minimap_obs()[5]
        if facing_direction == 0:  # down
            tile_infront = self.bottom_left_screen_tiles[5, 4]
        elif facing_direction == 4:  # up
            tile_infront = self.bottom_left_screen_tiles[3, 4]
        elif facing_direction == 8:  # left
            tile_infront = self.bottom_left_screen_tiles[4, 3]
        elif facing_direction == 12:  # right
            tile_infront = self.bottom_left_screen_tiles[4, 5]
        # tile_infront = self.read_ram_m(RAM.wTileInFrontOfPlayer)
        # no_collision = True
        use_surf_now = False
        if tile_infront in [0x14, 0x32, 0x48]:
            use_surf_now = True
            if tile_id == 17:
                # cavern
                # check for TilePairCollisionsWater
                tile_standingon = self.bottom_left_screen_tiles[4, 4]
                if tile_infront == 0x14 and tile_standingon == 5:
                    use_surf_now = False
            elif tile_id == 3:
                # forest
                # check for TilePairCollisionsWater
                tile_standingon = self.bottom_left_screen_tiles[4, 4]
                if tile_infront in [0x14, 0x48] and tile_standingon == 0x2e:
                    use_surf_now = False
            elif tile_id == 14:
                # vermilion dock
                # only 0x14 can be surfed on
                if tile_infront != 0x14:
                    use_surf_now = False
            elif tile_id == 13:
                # safari zone
                # only 0x14 can be surfed on
                if tile_infront != 0x14:
                    use_surf_now = False
        if use_surf_now:
            # temporary workaround
            map_id = self.current_map_id - 1
            x, y = self.current_coords
            if map_id == 8:
                map_name = MAP_ID_REF[map_id]
                map_width = MAP_DICT[map_name]['width']
                if ['CINNABAR_ISLAND', 'north'] in self.stage_manager.blockings and y == 0 and facing_direction == 4:
                    # skip
                    return
                elif ['CINNABAR_ISLAND', 'east'] in self.stage_manager.blockings and x == map_width - 1 and facing_direction == 12:
                    # skip
                    return
            self.pyboy.set_memory_value(RAM.wBattleAndStartSavedMenuItem.value, 1)
            self.run_action_on_emulator(action=10, emulated=WindowEvent.PRESS_BUTTON_START)
            self.run_action_on_emulator(action=4, emulated=WindowEvent.PRESS_BUTTON_A)
            for _ in range(3):
                self.pyboy.set_memory_value(RAM.wFieldMoves.value, 3)  # set first field move to surf
                self.pyboy.set_memory_value(RAM.wWhichPokemon.value, 0)  # first pokemon
                self.pyboy.set_memory_value(RAM.wMaxMenuItem.value, 3)  # max menu item
                self.run_action_on_emulator(action=4, emulated=WindowEvent.PRESS_BUTTON_A)
            # print(f'\nsurf used at step {self.step_count}, coords: {self.current_coords}, map: {MAP_ID_REF[self.current_map_id - 1]}')

    def scripted_routine_flute(self, action):
        if not self.can_use_flute:
            return
        # TURN THIS ON OR ELSE IT WILL AUTO FLUTE
        if action != 4:
            return
        
        if self.read_m(0xFFB0) == 0:
            # in menu
            return

        # can only be used in overworld
        tile_id = self.read_ram_m(RAM.wCurMapTileset)
        use_flute_now = False
        
        if tile_id in [0,]:
            minimap_sprite = self.get_minimap_sprite_obs()
            facing_direction = self.read_m(0xC109)  # wSpritePlayerStateData1FacingDirection
            if facing_direction == 0:  # down
                tile_infront = minimap_sprite[5, 4]
            elif facing_direction == 4:  # up
                tile_infront = minimap_sprite[3, 4]
            elif facing_direction == 8:  # left
                tile_infront = minimap_sprite[4, 3]
            elif facing_direction == 12:  # right
                tile_infront = minimap_sprite[4, 5]
            # tile_infront = self.read_ram_m(RAM.wTileInFrontOfPlayer)
            if tile_infront == 32:
                use_flute_now = True
        if use_flute_now:
            flute_bag_idx = self.get_items_in_bag().index(0x49)
            self.pyboy.set_memory_value(RAM.wBattleAndStartSavedMenuItem.value, 2)
            self.pyboy.set_memory_value(RAM.wBagSavedMenuItem.value, 0)
            self.pyboy.set_memory_value(RAM.wListScrollOffset.value, flute_bag_idx)
            self.run_action_on_emulator(action=10, emulated=WindowEvent.PRESS_BUTTON_START)
            self.run_action_on_emulator(action=4, emulated=WindowEvent.PRESS_BUTTON_A)
            for _ in range(10):  # help to skip through the wait
                self.run_action_on_emulator(action=4, emulated=WindowEvent.PRESS_BUTTON_A)

    def scripted_stage_blocking(self, action):    
        if not self.stage_manager.blockings:
            return action
        if self.read_m(0xFFB0) == 0:  # or action < 4
            # if not in menu, then check if we are blocked
            # if action is arrow, then check if we are blocked
            return action
        map_id = self.current_map_id - 1
        map_name = MAP_ID_REF[map_id]
        blocking_indexes = [idx for idx in range(len(self.stage_manager.blockings)) if self.stage_manager.blockings[idx][0] == map_name]
        # blocking_map_ids = [b[0] for b in self.stage_manager.blockings]
        if not blocking_indexes:
            return action
        x, y = self.current_coords
        new_x, new_y = x, y
        if action == 0:  # down
            new_y += 1
        elif action == 1:  # left
            new_x -= 1
        elif action == 2:  # right
            new_x += 1
        elif action == 3:  # up
            new_y -= 1
        # if new_x or new_y is blocked, then return noop button
        for idx in blocking_indexes:
            blocking = self.stage_manager.blockings[idx]
            blocked_dir_warp = blocking[1]
            if blocked_dir_warp in ['north', 'south', 'west', 'east']:
                if blocked_dir_warp == 'north' and action == 3 and new_y < 0:
                    return self.noop_button_index
                elif blocked_dir_warp == 'south' and action == 0 and new_y >= MAP_DICT[map_name]['height']:
                    return self.noop_button_index
                elif blocked_dir_warp == 'west' and action == 1 and new_x < 0:
                    return self.noop_button_index
                elif blocked_dir_warp == 'east' and action == 2 and new_x >= MAP_DICT[map_name]['width']:
                    return self.noop_button_index
            else:
                # blocked warp
                # get all warps in map
                warps = WARP_DICT[map_name]
                assert '@' in blocked_dir_warp, f'blocked_dir_warp: {blocked_dir_warp}'
                blocked_warp_map_name, blocked_warp_warp_id = blocked_dir_warp.split('@')
                for warp in warps:
                    if warp['target_map_name'] == blocked_warp_map_name and warp['warp_id'] == int(blocked_warp_warp_id):
                        if (new_x, new_y) == (warp['x'], warp['y']):
                            return self.noop_button_index
        return action
    
    def scripted_manage_party(self, action):
        # run scripted_party_management when pressing A facing the PC in pokecenter
        # indigo plateau lobby 0xAE
        pokecenter_map_ids = [0x29, 0x3A, 0x40, 0x44, 0x51, 0x59, 0x85, 0x8D, 0x9A, 0xAB, 0xB6, 0xAE, 0x81, 0xEB, 0xAA]
        map_id = self.current_map_id - 1
        if map_id not in pokecenter_map_ids:
            return action
        
        if action != 4:
            return action
        
        if self.read_m(0xFFB0) == 0:  # 0xFFB0 == 0 means in menu
            # in menu
            return action
        
        x, y = self.current_coords
        # 13, 4 is the coords below the pc
        # make sure we are facing the pc, up
        if map_id == 0xAE:
            if (x, y) != (15, 8):
                # for indigo plateau lobby, only do this when we are at 15, 8
                return action
        elif map_id == 0x81:
            # celadon mansion 2f
            if (x, y) != (0, 6):
                return action
        elif map_id == 0xEB:
            # silph co 11f
            if (x, y) != (10, 13):
                return action
        elif map_id == 0xAA:
            # cinnabar fossil room
            if (x, y) not in [(0, 5), (2, 5)]:
                return action
        elif (x, y) != (13, 4):
            return action
        
        # check if we are facing the pc
        facing_direction = self.read_m(0xC109)  # wSpritePlayerStateData1FacingDirection
        if facing_direction != 4:
            return action

        self.scripted_party_management()

        return self.noop_button_index
    
    def can_auto_press_a(self):
        if self.read_m(0xc4f2) == 238 and \
            self.read_m(0xFFB0) == 0 and \
                self.read_ram_m(RAM.wTextBoxID) == 1 and \
                    self.read_m(0xFF8B) != 0:  # H_DOWNARROWBLINKCNT1
            return True
        else:
            return False

    def run_action_on_emulator(self, action, emulated=0):
        if not self.read_ram_bit(RAM.wd730, 6):
            # if not instant text speed, then set it to instant
            txt_value = self.read_ram_m(RAM.wd730)
            self.pyboy.set_memory_value(RAM.wd730.value, self.set_bit(txt_value, 6))
        if self.enable_stage_manager and action < 4:
            # enforce stage_manager.blockings
            action = self.scripted_stage_blocking(action)
        if self.enable_item_purchaser and self.current_map_id - 1 in MART_MAP_IDS and action == 4:
            can_buy = self.scripted_buy_items()
            if can_buy:
                action = self.noop_button_index
        if not emulated and self.extra_buttons and self.restricted_start_menu:
            # restrict start menu choices
            action = self.get_menu_restricted_action(action)
        # press button then release after some steps
        if not emulated:
            if action == 4:
                self.scripted_routine_flute(action)
                self.scripted_routine_cut(action)
                self.scripted_routine_surf(action)
                action = self.scripted_manage_party(action)
            self.pyboy.send_input(self.valid_actions[action])
        else:
            self.pyboy.send_input(emulated)
        # disable rendering when we don't need it
        # if self.headless and (self.fast_video or not self.save_video):
        if emulated or (self.headless and (self.fast_video or not self.save_video)):
            self.pyboy._rendering(False)
        for i in range(self.act_freq):
            # release action, so they are stateless
            if i == 8:
                if action < 4:
                    # release arrow
                    self.pyboy.send_input(self.release_arrow[action])
                if action > 3 and action < 6:
                    # release button 
                    self.pyboy.send_input(self.release_button[action - 4])
                if not emulated and self.valid_actions[action] == WindowEvent.PRESS_BUTTON_START:
                    self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
                elif emulated and emulated == WindowEvent.PRESS_BUTTON_START:
                    self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
            if self.save_video and not self.fast_video:
                self.add_video_frame()
            if i == self.act_freq-1 and not emulated and not self.can_auto_press_a():
                self.pyboy._rendering(True)
            self.pyboy.tick()
        if self.save_video and self.fast_video:
            self.add_video_frame()
    
    def add_video_frame(self):
        self.full_frame_writer.add_image(self.render(reduce_res=False, update_mem=False))
        # self.model_frame_writer.add_image(self.render(reduce_res=True, update_mem=False))

    # @property
    # def last_episode_stats(self):
    #     if not self._last_episode_stats:
    #         return self.get_stats()
    #     return self._last_episode_stats

    @property
    def current_stats_with_id(self):
        stats = self.get_stats()
        stats['env_id'] = self.env_id
        return stats
    
    def get_stats(self):
        stats = {
            'step': self.step_count,
            'rewards': self.total_reward,
            'eventr': self.progress_reward['event'],
            'levelr': self.progress_reward['level'],
            'op_lvlr': self.progress_reward['op_lvl'],
            'deadr': self.progress_reward['dead'],
            'visited_pokecenterr': self.progress_reward['visited_pokecenter'],
            'trees_cutr': self.progress_reward['trees_cut'],
            'hmr': self.progress_reward['hm'],
            'hm_usabler': self.progress_reward['hm_usable'],
            'special_key_itemsr': self.progress_reward['special_key_items'],
            'special_seen_coords_count': self.special_seen_coords_count,
            'specialr': self.progress_reward['special'],
            'coord_count': len(self.seen_coords),
            'perm_coord_count': len(self.perm_seen_coords),
            'seen_map_count': len(self.seen_map_dict),
            'hp': self.last_health,  # use last health to avoid performance hit
            'pcount': self.last_num_poke,  # use last num poke to avoid performance hit
            'healr': self.progress_reward['heal'],
            # 'es_min_reward': self.early_stopping_min_reward,
            'n_pc_swap': self.use_pc_swap_count,
            'n_buy': self.use_mart_count,
            'current_level': self.current_level,
            'badger': self.progress_reward['badge'],
            'level_sum': self.get_levels_sum(),
            'level_post': self.party_level_post,
        }
        if self.enable_stage_manager:
            stats['n_stage'] = self.stage_manager.stage
            stats['stager'] = self.progress_reward['stage']

        badges_bin = bin(self.read_ram_m(RAM.wObtainedBadges))[2:].zfill(8)
        badges = [int(b) for b in badges_bin]
        for i, badge in enumerate(badges):
            if badge:
                stats[f'badge_{8-i}'] = 1
        
        for pokecenter_id in self.visited_pokecenter_list:
            # idx = self.pokecenter_ids.index(pokecenter_id)
            stats[f'pokecenter_{pokecenter_id}'] = 1

        # commented out for now, too overwhleming
        # seen_map_ids = list(self.seen_map_dict.keys())
        # for seen_map_id in seen_map_ids:
        #     map_name = MAP_ID_REF[seen_map_id]
        #     stats[f'map_{int(seen_map_id)}_{map_name}'] = 1
        return stats

    @property
    def last_agent_stats(self):
        if self.agent_stats:
            return self.agent_stats[-1]
        else:
            return {}
    
    # def append_agent_stats(self, action):
    #     x_pos = self.read_m(0xD362)
    #     y_pos = self.read_m(0xD361)
    #     map_n = self.read_m(0xD35E)
    #     levels = [self.read_m(a) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]]
    #     # if self.use_screen_explore:
    #     #     expl = ('frames', self.knn_index.get_current_count())
    #     # else:
    #     expl = ('coord_count', len(self.seen_coords))
    #     self.agent_stats.append({
    #         'step': self.step_count, 'x': x_pos, 'y': y_pos, 'map': map_n,
    #         'last_action': action,
    #         'pcount': self.read_m(0xD163), 
    #         'levels': levels, 
    #         'ptypes': self.read_party(),
    #         'hp': self.read_hp_fraction(),
    #         expl[0]: expl[1],
    #         'prev_knn_rew': self.prev_knn_rew,
    #         # 'deaths': self.died_count, 
    #         'badge': self.get_badges(),
    #         'eventr': self.progress_reward['event'],
    #         'levelr': self.progress_reward['level'],
    #         'op_lvlr': self.progress_reward['op_lvl'],
    #         'deadr': self.progress_reward['dead'],
    #         'visited_pokecenterr': self.progress_reward['visited_pokecenter'],
    #         'trees_cutr': self.progress_reward['trees_cut'],
    #         'hmr': self.progress_reward['hm'],
    #         'hm_usabler': self.progress_reward['hm_usable'],
    #         # 'hm_mover': self.progress_reward['hm_move'],
    #         'rewards': self.total_reward,
    #         # 'early_done': self.early_done,
    #         'special_key_itemsr': self.progress_reward['special_key_items'],
    #         'special_seen_coords_count': self.special_seen_coords_count,
    #     })
    
    def update_seen_coords(self):
        x_pos, y_pos = self.current_coords
        map_n = self.current_map_id - 1
        coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"
        if self.special_exploration_scale and map_n in SPECIAL_MAP_IDS and coord_string not in self.perm_seen_coords:
            # self.seen_coords[coord_string] = self.step_count
            self.special_seen_coords_count += 1
        self.seen_coords[coord_string] = self.step_count
        self.perm_seen_coords[coord_string] = self.step_count

    def update_reward(self):
        # compute reward
        # old_prog = self.group_rewards()
        self.progress_reward = self.get_game_state_reward()
        # new_prog = self.group_rewards()
        new_total = sum([val for _, val in self.progress_reward.items()]) #sqrt(self.explore_reward * self.progress_reward)
        new_step = new_total - self.total_reward
        # if new_step < 0 and self.read_hp_fraction() > 0:
        #     #print(f'\n\nreward went down! {self.progress_reward}\n\n')
        #     self.save_screenshot('neg_reward')
    
        self.total_reward = new_total
        return new_step
    
    def group_rewards(self):
        prog = self.progress_reward
        # these values are only used by memory
        return (prog['level'] * 100 / self.reward_scale, 
                self.read_hp_fraction()*2000, 
                prog['explore'] * 150 / (self.explore_weight * self.reward_scale))
               #(prog['events'], 
               # prog['levels'] + prog['party_xp'], 
               # prog['explore'])

    def create_exploration_memory(self):
        w = self.output_shape[1]
        h = self.memory_height
        
        def make_reward_channel(r_val):
            col_steps = self.col_steps
            max_r_val = (w-1) * h * col_steps
            # truncate progress bar. if hitting this
            # you should scale down the reward in group_rewards!
            r_val = min(r_val, max_r_val)
            row = floor(r_val / (h * col_steps))
            memory = np.zeros(shape=(h, w), dtype=np.uint8)
            memory[:, :row] = 255
            row_covered = row * h * col_steps
            col = floor((r_val - row_covered) / col_steps)
            memory[:col, row] = 255
            col_covered = col * col_steps
            last_pixel = floor(r_val - row_covered - col_covered) 
            memory[col, row] = last_pixel * (255 // col_steps)
            return memory
        
        level, hp, explore = self.group_rewards()
        full_memory = np.stack((
            make_reward_channel(level),
            make_reward_channel(hp),
            make_reward_channel(explore)
        ), axis=-1)
        
        if self.get_badges() > 0:
            full_memory[:, -1, :] = 255

        return full_memory

    # def create_recent_memory(self):
    #     return rearrange(
    #         self.recent_memory, 
    #         '(w h) c -> h w c', 
    #         h=self.memory_height)
    
    def check_if_early_done(self):
        # self.early_done = False
        if self.early_stopping and self.step_count > 10239:
            # early stop if less than 500 new coords or 125 special coords or any rewards lesser than 2, like so
            # 2 events, 1 new pokecenter, 2 level, 8 trees cut, 1 hm(+event), 1 hm usable, 1 badge, 2 special_key_items, 1 special reward
            # if 4
            # 1,000 new coords or 250 special coords
            # 3 events, 1 new pokecenter, 3 level, 12 trees cut, 1 hm(+event), 1 hm usable, 1 badge, 3 special_key_items, 1.5 special reward
            # if self.stage_manager.stage == 11 and self.current_map_id - 1 in [0xF5, 0xF6, 0xF7, 0x71, 0x78] and self.elite_4_started_step is not None and self.step_count - self.elite_4_started_step > 1600:
            #     # if in elite 4 rooms
            #     self.early_done = self.past_rewards[0] - self.past_rewards[1600] < (self.early_stopping_min_reward / 4 * self.reward_scale)
            #     if self.early_done:
            #         num_badges = self.get_badges()
            #         print(f'elite 4 early done, step: {self.step_count}, r1: {self.past_rewards[0]:6.2f}, r2: {self.past_rewards[1600]:6.2f}, badges: {num_badges}')
            #         self.elite_4_early_done = True
            # else:
            self.early_done = self.past_rewards[0] - self.past_rewards[-1] < (self.early_stopping_min_reward * self.reward_scale)
            if self.early_done:
                if self.elite_4_early_done:
                    num_badges = self.get_badges()
                    print(f'elite 4 early done, step: {self.env_id}:{self.step_count}, r1: {self.past_rewards[0]:6.2f}, r2: {self.past_rewards[1600]:6.2f}, badges: {num_badges}')
                    self.elite_4_early_done = True
                else:
                    print(f'es, step: {self.env_id}:{self.step_count}, r1: {self.past_rewards[0]:6.2f}, r2: {self.past_rewards[-1]:6.2f}')
        return self.early_done

    def check_if_done(self):
        done = self.step_count >= self.max_steps
        if self.early_done:
            done = True
        return done

    def save_and_print_info(self, done, obs_memory):
        if self.print_rewards:
            if self.step_count % 5 == 0:
                prog_string = f's: {self.step_count:7d} env: {self.current_level:1}:{self.env_id:2}'
                if self.enable_stage_manager:
                    prog_string += f' stage: {self.stage_manager.stage:2d}'
                for key, val in self.progress_reward.items():
                    if key in ['level', 'explore', 'event', 'dead']:
                        prog_string += f' {key}: {val:6.2f}'
                    elif key in ['level_completed', 'early_done']:
                        continue
                    else:
                        prog_string += f' {key[:10]}: {val:5.2f}'
                prog_string += f' sum: {self.total_reward:5.2f}'
                print(f'\r{prog_string}', end='', flush=True)
        
        if self.step_count % 1000 == 0:
            try:
                plt.imsave(
                    self.s_path / Path(f'curframe_{self.env_id}.jpeg'), 
                    self.render(reduce_res=False))
            except:
                pass

        if self.print_rewards and done:
            print('', flush=True)
            if self.save_final_state:
                fs_path = self.s_path / Path('final_states')
                fs_path.mkdir(exist_ok=True)
                try:
                    # plt.imsave(
                    #     fs_path / Path(f'frame_r{self.total_reward:.4f}_{self.reset_count}_small.jpeg'), 
                    #     rearrange(obs_memory['image'], 'c h w -> h w c'))
                    plt.imsave(
                        fs_path / Path(f'frame_r{self.total_reward:.4f}_{self.reset_count}_full.jpeg'), 
                        self.render(reduce_res=False))
                except Exception as e:
                    print(f'error saving final state: {e}')
                # if self.save_state_dir:
                #     self.save_all_states()

        if self.save_video and done:
            self.full_frame_writer.close()
            # modify video name to include final reward (self.total_reward) as prefix
            new_name = f'r{self.total_reward:.4f}_env{self.env_id}_{self.reset_count}.mp4'
            new_path = self.full_frame_write_full_path.parent / Path(new_name)
            self.full_frame_write_full_path.rename(new_path)
            # self.model_frame_writer.close()
        
        if self.save_state_dir:
            if done:
                if self.level_completed:
                    self.save_all_states()
                elif not self.early_done:
                    # do not save early done at all, useless info
                    self.save_all_states(is_failed=True)
                self.record_statistic()
            elif self.level_completed and self.level_manager_eval_mode:
                self.save_all_states()
                self.record_statistic()

        if done:
            self.all_runs.append(self.progress_reward)
            with open(self.s_path / Path(f'all_runs_{self.env_id}.json'), 'w') as f:
                json.dump(self.all_runs, f)
            pd.DataFrame(self.agent_stats).to_csv(
                self.s_path / Path(f'agent_stats_{self.env_id}.csv.gz'), compression='gzip', mode='a')
    
    def record_statistic(self):
        if self.save_state_dir:
            stats_path = self.save_state_dir / Path('stats')
            stats_path.mkdir(exist_ok=True)
            with open(stats_path / Path(f'level_{self.current_level}.txt'), 'a') as f:
                # append S for success and F for failure
                if self.level_completed:
                    f.write(f'S')
                elif self.early_done:
                    f.write(f'F')
                else:
                    f.write(f'F')

    def read_m(self, addr):
        return self.pyboy.get_memory_value(addr)

    def read_bit(self, addr, bit: int) -> bool:
        # add padding so zero will read '0b100000000' instead of '0b0'
        return bin(256 + self.read_m(addr))[-bit-1] == '1'
    
    def read_ram_m(self, addr: RAM) -> int:
        return self.pyboy.get_memory_value(addr.value)
    
    def read_ram_bit(self, addr: RAM, bit: int) -> bool:
        return bin(256 + self.read_ram_m(addr))[-bit-1] == '1'
    
    def get_levels_sum(self):
        poke_levels = [max(self.read_m(a) - 2, 0) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]]
        return max(sum(poke_levels) - 4, 0) # subtract starting pokemon level
    
    def get_max_n_levels_sum(self, n, max_level):
        num_poke = self.read_num_poke()
        poke_level_addresses = [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
        poke_levels = [max(min(self.read_m(a), max_level) - 2, 0) for a in poke_level_addresses[:num_poke]]
        return max(sum(sorted(poke_levels)[-n:]) - 4, 0)
    
    def get_levels_reward(self):
        if not self.level_reward_badge_scale:
            level_sum = self.get_levels_sum()
            self.max_level_rew = max(self.max_level_rew, level_sum)
        else:
            badge_count = min(self.get_badges(), 8)
            gym_next = self.gym_info[badge_count]
            gym_num_poke = gym_next['num_poke']
            gym_max_level = gym_next['max_level'] * self.level_reward_badge_scale
            level_reward = self.get_max_n_levels_sum(gym_num_poke, gym_max_level)  # changed, level reward for all 6 pokemon
            if badge_count >= 7 and level_reward > self.max_level_rew and not self.is_in_elite_4:
                level_diff = level_reward - self.max_level_rew
                if level_diff > 6 and self.party_level_post == 0:
                    # self.party_level_post = 0
                    pass
                else:
                    self.party_level_post += level_diff
            self.max_level_rew = max(self.max_level_rew, level_reward)
        return ((self.max_level_rew - self.party_level_post) * 0.5) + (self.party_level_post * 2.0)
        # return self.max_level_rew * 0.5  # 11/11-3 changed: from 0.5 to 1.0
    
    @property
    def is_in_elite_4(self):
        return self.current_map_id - 1 in [0xF5, 0xF6, 0xF7, 0x71, 0x78]
    
    def get_early_done_reward(self):
        return self.elite_4_early_done * -0.3
    
    def get_knn_reward(self, last_event_rew):
        seen_coord_scale = 0.5
        perm_seen_coord_scale = 0.5
        knn_reward_scale = 0.005
        if self.special_exploration_scale:
            special_count = self.special_seen_coords_count * self.special_exploration_scale
        else:
            special_count = 0
        if last_event_rew != self.max_event_rew:
            # event reward increased, reset exploration
            self.prev_knn_rew += len(self.seen_coords)
            self.seen_coords = {}
        cur_size = len(self.seen_coords)
        # cur_size = self.knn_index.get_current_count() if self.use_screen_explore else len(self.seen_coords)
        return (
            ((self.prev_knn_rew + cur_size) * seen_coord_scale) + \
            (len(self.perm_seen_coords) * perm_seen_coord_scale) + \
                special_count) * self.explore_weight * knn_reward_scale
    
    def get_knn_reward_exclusion(self):
        # exclude prev_knn_rew and cur_size
        seen_coord_scale = 0.5
        knn_reward_scale = 0.005
        cur_size = len(self.seen_coords)
        return ((self.prev_knn_rew + cur_size) * seen_coord_scale) * self.explore_weight * knn_reward_scale
    
    def update_visited_pokecenter_list(self):
        last_pokecenter_id = self.get_last_pokecenter_id()
        if last_pokecenter_id != -1 and last_pokecenter_id not in self.visited_pokecenter_list:
            self.visited_pokecenter_list.append(last_pokecenter_id)

    def get_visited_pokecenter_reward(self):
        # reward for first time healed in pokecenter
        return len(self.visited_pokecenter_list) * 2
    
    def get_badges_reward(self):
        num_badges = self.get_badges()
        # if num_badges < 3:
        #     return num_badges * 5
        # elif num_badges > 2:
        #     return 10 + ((num_badges - 2) * 10)  # reduced from 20 to 10
        if num_badges < 9:
            return num_badges * 5
        elif num_badges < 13:  # env19v2 PPO23
            return 40  # + ((num_badges - 8) * 1)
        else:
            return 40 + 10
        # return num_badges * 5  # env18v4
    
    def get_badges(self):
        badge_count = self.bit_count(self.read_m(0xD356))
        # return badge_count
        if badge_count < 8 or self.elite_4_lost or self.elite_4_early_done:
            return badge_count
        else:
            # LORELEIS D863, bit 1
            # BRUNOS D864, bit 1
            # AGATHAS D865, bit 1
            # LANCES D866, bit 1
            # CHAMPION D867, bit 1
            elite_four_event_addr_bits = [
                [0xD863, 1],  # LORELEIS
                [0xD864, 1],  # BRUNOS
                [0xD865, 1],  # AGATHAS
                [0xD866, 1],  # LANCES
                [0xD867, 1],  # CHAMPION
            ]
            elite_4_extra_badges = 0
            for addr_bit in elite_four_event_addr_bits:
                if self.read_bit(addr_bit[0], addr_bit[1]):
                    elite_4_extra_badges += 1
            return 8 + elite_4_extra_badges


    def read_party(self, one_indexed=0):
        parties = [self.read_m(addr) for addr in [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]]
        return [p + one_indexed if p != 0xff and p != 0 else 0 for p in parties]
    
    def get_last_pokecenter_list(self):
        pc_list = [0, ] * len(self.pokecenter_ids)
        last_pokecenter_id = self.get_last_pokecenter_id()
        if last_pokecenter_id != -1:
            pc_list[last_pokecenter_id] = 1
        return pc_list
    
    def get_last_pokecenter_id(self):
        
        last_pokecenter = self.read_m(0xD719)
        # will throw error if last_pokecenter not in pokecenter_ids, intended
        if last_pokecenter == 0:
            # no pokecenter visited yet
            return -1
        if last_pokecenter not in self.pokecenter_ids:
            print(f'\nERROR: last_pokecenter: {last_pokecenter} not in pokecenter_ids')
            return -1
        else:
            return self.pokecenter_ids.index(last_pokecenter)
    
    def get_hm_rewards(self):
        hm_ids = [0xC4, 0xC5, 0xC6, 0xC7, 0xC8]
        items = self.get_items_in_bag()
        total_hm_cnt = 0
        for hm_id in hm_ids:
            if hm_id in items:
                total_hm_cnt += 1
        return total_hm_cnt * 1

    def update_heal_reward(self):
        cur_health = self.read_hp_fraction()
        if cur_health > self.last_health:
            # fixed catching pokemon might treated as healing
            # fixed leveling count as healing, min heal amount is 4%
            heal_amount = cur_health - self.last_health
            if self.last_num_poke == self.read_num_poke() and self.last_health > 0 and heal_amount > 0.04:
                if heal_amount > (0.60 / self.read_num_poke()):
                    # changed to static heal reward
                    # 1 pokemon from 0 to 100% hp is 0.167 with 6 pokemon
                    # so 0.1 total heal is around 60% hp
                    print(f' healed: {heal_amount:.2f}')
                    self.total_healing_rew += 0.1
                # if heal_amount > 0.5:
                #     print(f' healed: {heal_amount:.2f}')
                #     # self.save_screenshot('healing')
                # self.total_healing_rew += heal_amount * 1
            elif self.last_health <= 0:
                    self.died_count += 1

    def update_num_poke(self):
        self.last_num_poke = self.read_num_poke()

    def update_num_mon_in_box(self):
        self.last_num_mon_in_box = self.num_mon_in_box

    def get_base_event_flags(self):
        # event patches
        # 1. triggered EVENT_FOUND_ROCKET_HIDEOUT 
        # event_value = self.read_m(0xD77E)  # bit 1
        # self.pyboy.set_memory_value(0xD77E, self.set_bit(event_value, 1))
        # 2. triggered EVENT_GOT_TM13 , fresh_water trade
        event_value = self.read_m(0xD778)  # bit 4
        self.pyboy.set_memory_value(0xD778, self.set_bit(event_value, 4))
        address_bits = [
            # seafoam islands
            [0xD7E8, 6],
            [0xD7E8, 7],
            [0xD87F, 0],
            [0xD87F, 1],
            [0xD880, 0],
            [0xD880, 1],
            [0xD881, 0],
            [0xD881, 1],
            # victory road
            [0xD7EE, 0],
            [0xD7EE, 7],
            [0xD813, 0],
            [0xD813, 6],
            [0xD869, 7],
        ]
        for ab in address_bits:
            event_value = self.read_m(ab[0])
            self.pyboy.set_memory_value(ab[0], self.set_bit(event_value, ab[1]))

        n_ignored_events = 0
        for event_id in IGNORED_EVENT_IDS:
            if self.all_events_string[event_id] == '1':
                n_ignored_events += 1
        return max(
            self.all_events_string.count('1')
            - n_ignored_events,
        0,
    )

    # def get_all_events_reward(self):
    #     # adds up all event flags, exclude museum ticket
    #     # museum_ticket = (0xD754, 0)
    #     # base_event_flags = 13
    #     n_ignored_events = 0
    #     for event_id in IGNORED_EVENT_IDS:
    #         if self.all_events_string[event_id] == '1':
    #             n_ignored_events += 1
    #     return max(
    #         self.all_events_string.count('1')
    #         - self.base_event_flags
    #         - n_ignored_events,
    #     0,
    # )
    
    def get_all_events_reward(self):
        if self.all_events_string != self.past_events_string:
            first_i = -1
            for i in range(len(self.all_events_string)):
                if self.all_events_string[i] == '1' and self.rewarded_events_string[i] == '0' and i not in IGNORED_EVENT_IDS:
                    self.rewarded_events_string = self.rewarded_events_string[:i] + '1' + self.rewarded_events_string[i+1:]
                    if first_i == -1:
                        first_i = i
            if first_i != -1:
                # update past event ids
                self.last_10_event_ids = np.roll(self.last_10_event_ids, 1, axis=0)
                self.last_10_event_ids[0] = [first_i, self.step_count]
        if self.stage_manager.stage != 11:
            return self.rewarded_events_string.count('1') - self.base_event_flags
        else:
            # elite 4 stage
            elite_four_event_addr_bits = [
                [0xD863, 0],  # EVENT START
                [0xD863, 1],  # LORELEIS
                [0xD863, 6],  # LORELEIS AUTO WALK
                [0xD864, 1],  # BRUNOS
                [0xD864, 6],  # BRUNOS AUTO WALK
                [0xD865, 1],  # AGATHAS
                [0xD865, 6],  # AGATHAS AUTO WALK
                [0xD866, 1],  # LANCES
                [0xD866, 6],  # LANCES AUTO WALK
            ]
            ignored_elite_four_events = 0
            for ab in elite_four_event_addr_bits:
                if self.get_event_rewarded_by_address(ab[0], ab[1]):
                    ignored_elite_four_events += 1
            return self.rewarded_events_string.count('1') - self.base_event_flags - ignored_elite_four_events
        

    def get_game_state_reward(self, print_stats=False):
        # addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm
        '''
        num_poke = self.read_m(0xD163)
        poke_xps = [self.read_triple(a) for a in [0xD179, 0xD1A5, 0xD1D1, 0xD1FD, 0xD229, 0xD255]]
        #money = self.read_money() - 975 # subtract starting money
        seen_poke_count = sum([self.bit_count(self.read_m(i)) for i in range(0xD30A, 0xD31D)])
        all_events_score = sum([self.bit_count(self.read_m(i)) for i in range(0xD747, 0xD886)])
        oak_parcel = self.read_bit(0xD74E, 1) 
        oak_pokedex = self.read_bit(0xD74B, 5)
        opponent_level = self.read_m(0xCFF3)
        self.max_opponent_level = max(self.max_opponent_level, opponent_level)
        enemy_poke_count = self.read_m(0xD89C)
        self.max_opponent_poke = max(self.max_opponent_poke, enemy_poke_count)
        
        if print_stats:
            print(f'num_poke : {num_poke}')
            print(f'poke_levels : {poke_levels}')
            print(f'poke_xps : {poke_xps}')
            #print(f'money: {money}')
            print(f'seen_poke_count : {seen_poke_count}')
            print(f'oak_parcel: {oak_parcel} oak_pokedex: {oak_pokedex} all_events_score: {all_events_score}')
        '''
        last_event_rew = self.max_event_rew
        self.max_event_rew = self.update_max_event_rew()
        state_scores = {
            'event': self.max_event_rew,  
            #'party_xp': self.reward_scale*0.1*sum(poke_xps),
            'level': self.get_levels_reward(), 
            # 'heal': self.total_healing_rew,
            'op_lvl': self.update_max_op_level(),
            'dead': -self.get_dead_reward(),
            'badge': self.get_badges_reward(),  # 5
            #'op_poke':self.max_opponent_poke * 800,
            #'money': money * 3,
            #'seen_poke': self.reward_scale * seen_poke_count * 400,
            'explore': self.get_knn_reward(last_event_rew),
            'visited_pokecenter': self.get_visited_pokecenter_reward(),
            'hm': self.get_hm_rewards(),
            # 'hm_move': self.get_hm_move_reward(),  # removed this for now
            'hm_usable': self.get_hm_usable_reward(),
            'trees_cut': self.get_used_cut_coords_reward(),
            'early_done': self.get_early_done_reward(),  # removed
            'special_key_items': self.get_special_key_items_reward(),
            'special': self.get_special_rewards(),
            'heal': self.total_healing_rew,
            'level_completed': self.get_level_completed_reward(),
        }
        if self.enable_stage_manager:
            state_scores['stage'] = self.get_stage_rewards()
        # multiply by reward scale
        state_scores = {k: v * self.reward_scale for k, v in state_scores.items()}
        
        return state_scores
    
    def get_dead_reward(self):
        # money_weight = np.clip(self.read_money() / 100_000.0, 0.1, 1.0)
        # return -money_weight * self.died_count
        return 0.1 * self.died_count  # modified from 0.1 to 0.3 after 400k+ steps
    
    def get_level_completed_reward(self):
        # if self.level_completed:
        #     return 5.0
        # return 0.0
        if self.level_completed:
            # to make sure non eval mode got the completed rewards before ending
            completed_levels = self.current_level + 1
        else:
            completed_levels = self.current_level
        return completed_levels * 5.0
    
    def get_stage_rewards(self):
        return self.stage_manager.n_stage_started * 1.0 + self.stage_manager.n_stage_ended * 1.0
    
    def get_special_rewards(self):
        rewards = 0
        rewards += len(self.hideout_elevator_maps) * 2.0
        bag_items = self.get_items_in_bag()
        if 0x2B in bag_items:
            # 6.0 full mansion rewards + 1.0 extra key items rewards
            rewards += 7.0
        elif self.stage_manager.stage >= 10:
            map_id = self.current_map_id - 1
            mansion_rewards = 0
            if map_id == 0xD8:
                # pokemon mansion b1f
                mansion_rewards += 4.0
                if 'given_reward' in self.secret_switch_states:
                    mansion_rewards += self.secret_switch_states['given_reward']
            # max mansion_rewards is 12.0 * 0.5 = 6.0 actual rewards
            rewards += mansion_rewards * 0.5
        return rewards
    
    def get_hm_usable_reward(self):
        total = 0
        if self.can_use_cut:
            total += 1
        if self.can_use_surf:
            total += 1
        return total * 2.0
    
    def get_special_key_items_reward(self):
        items = self.get_items_in_bag()
        special_cnt = 0
        # SPECIAL_KEY_ITEM_IDS
        for item_id in SPECIAL_KEY_ITEM_IDS:
            if item_id in items:
                special_cnt += 1
        return special_cnt * 1.0
    
    def get_used_cut_coords_reward(self):
        return len(self.used_cut_coords_dict) * 0.2
    
    def get_hm_move_reward(self):
        all_moves = self.get_party_moves()
        hm_moves = [0x0f, 0x13, 0x39, 0x46, 0x94]
        hm_move_count = 0
        for hm_move in hm_moves:
            if hm_move in all_moves:
                hm_move_count += 1
        return hm_move_count * 1.5
    
    def save_screenshot(self, name):
        ss_dir = self.s_path / Path('screenshots')
        ss_dir.mkdir(exist_ok=True)
        uuid_str = uuid.uuid4().hex
        plt.imsave(
            ss_dir / Path(f'frame{self.env_id}_r{self.total_reward:.4f}_{self.reset_count}_{name}_{str(uuid_str)[:4]}.jpeg'),
            self.render(reduce_res=False))
    
    def update_max_op_level(self):
        #opponent_level = self.read_m(0xCFE8) - 5 # base level
        opponent_level = max([self.read_m(a) for a in [0xD8C5, 0xD8F1, 0xD91D, 0xD949, 0xD975, 0xD9A1]]) - 5
        #if opponent_level >= 7:
        #    self.save_screenshot('highlevelop')
        self.max_opponent_level = max(self.max_opponent_level, opponent_level)
        return self.max_opponent_level * 0.1  # 0.1
    
    @property
    def all_events_string(self):
        # cache all events string to improve performance
        if not self._all_events_string:
            event_flags_start = 0xD747
            event_flags_end = 0xD886
            result = ''
            for i in range(event_flags_start, event_flags_end):
                result += bin(self.read_m(i))[2:].zfill(8)  # .zfill(8)
            self._all_events_string = result
        return self._all_events_string
    
    def get_event_rewarded_by_address(self, address, bit):
        # read from rewarded_events_string
        event_flags_start = 0xD747
        event_pos = address - event_flags_start
        # bit is reversed
        # string_pos = event_pos * 8 + bit
        string_pos = event_pos * 8 + (7 - bit)
        return self.rewarded_events_string[string_pos] == '1'
    
    @property
    def battle_type(self):
        if self._battle_type is None:
            result = self.read_m(0xD057)
            if result == -1:
                self._battle_type = 0
            else:
                self._battle_type = result
        return self._battle_type
    
    def is_wild_battle(self):
        return self.battle_type == 1
    
    def update_max_event_rew(self):
        if self.all_events_string != self.past_events_string:
            cur_rew = self.get_all_events_reward()
            self.max_event_rew = max(cur_rew, self.max_event_rew)
        return self.max_event_rew
    
    def is_in_battle(self):
        # D057
        # 0 not in battle
        # 1 wild battle
        # 2 trainer battle
        # -1 lost battle
        return self.battle_type > 0
    
    def get_items_in_bag(self, one_indexed=0):
        if self._items_in_bag is None:
            first_item = 0xD31E
            # total 20 items
            # item1, quantity1, item2, quantity2, ...
            item_ids = []
            for i in range(0, 40, 2):
                item_id = self.read_m(first_item + i)
                if item_id == 0 or item_id == 0xff:
                    break
                item_ids.append(item_id)
            self._items_in_bag = item_ids
        else:
            item_ids = self._items_in_bag
        if one_indexed:
            return [i + 1 for i in item_ids]
        return item_ids
    
    def get_items_quantity_in_bag(self):
        first_quantity = 0xD31E
        # total 20 items
        # quantity1, item2, quantity2, ...
        item_quantities = []
        for i in range(1, 40, 2):
            item_quantity = self.read_m(first_quantity + i)
            if item_quantity == 0 or item_quantity == 0xff:
                break
            item_quantities.append(item_quantity)
        return item_quantities
    
    def get_party_moves(self):
        # first pokemon moves at D173
        # 4 moves per pokemon
        # next pokemon moves is 44 bytes away
        first_move = 0xD173
        moves = []
        for i in range(0, 44*6, 44):
            # 4 moves per pokemon
            move = [self.read_m(first_move + i + j) for j in range(4)]
            moves.extend(move)
        return moves

    def read_hp_fraction(self):
        hp_sum = sum([self.read_hp(add) for add in [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]])
        max_hp_sum = sum([self.read_hp(add) for add in [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]])
        if max_hp_sum:
            return hp_sum / max_hp_sum
        else:
            return 0

    def read_hp(self, start):
        return 256 * self.read_m(start) + self.read_m(start+1)

    # built-in since python 3.10
    def bit_count(self, bits):
        return bin(bits).count('1')

    def read_triple(self, start_add):
        return 256*256*self.read_m(start_add) + 256*self.read_m(start_add+1) + self.read_m(start_add+2)
    
    def read_bcd(self, num):
        return 10 * ((num >> 4) & 0x0f) + (num & 0x0f)
    
    def read_double(self, start_add):
        return 256*self.read_m(start_add) + self.read_m(start_add+1)
    
    def read_money(self):
        return (100 * 100 * self.read_bcd(self.read_m(0xD347)) + 
                100 * self.read_bcd(self.read_m(0xD348)) +
                self.read_bcd(self.read_m(0xD349)))

    def read_num_poke(self):
        return self.read_m(0xD163)
    
    def multi_hot_encoding(self, cnt, max_n):
        return [1 if cnt < i else 0 for i in range(max_n)]
    
    def one_hot_encoding(self, cnt, max_n, start_zero=False):
        if start_zero:
            return [1 if cnt == i else 0 for i in range(max_n)]
        else:
            return [1 if cnt == i+1 else 0 for i in range(max_n)]
    
    def scaled_encoding(self, cnt, max_n: float):
        max_n = float(max_n)
        if isinstance(cnt, list):
            return [min(1.0, c / max_n) for c in cnt]
        elif isinstance(cnt, np.ndarray):
            return np.clip(cnt / max_n, 0, 1)
        else:
            return min(1.0, cnt / max_n)
    
    def get_badges_obs(self):
        return self.multi_hot_encoding(self.get_badges(), 12)

    def get_money_obs(self):
        return [self.scaled_encoding(self.read_money(), 100_000)]
    
    def read_swap_mon_pos(self):
        is_in_swap_mon_party_menu = self.read_m(0xd07d) == 0x04
        if is_in_swap_mon_party_menu:
            chosen_mon = self.read_m(0xcc35)
            if chosen_mon == 0:
                print(f'\nsomething went wrong, chosen_mon is 0')
            else:
                return chosen_mon - 1
        return -1
    
    # def get_swap_pokemon_obs(self):
    #     is_in_swap_mon_party_menu = self.read_m(0xd07d) == 0x04
    #     if is_in_swap_mon_party_menu:
    #         chosen_mon = self.read_m(0xcc35)
    #         if chosen_mon == 0:
    #             print(f'\nsomething went wrong, chosen_mon is 0')
    #         else:
    #             # print(f'chose mon {chosen_mon}')
    #             return self.one_hot_encoding(chosen_mon - 1, 6, start_zero=True)
    #     return [0] * 6
    
    def get_last_pokecenter_obs(self):
        return self.get_last_pokecenter_list()

    def get_visited_pokecenter_obs(self):
        result = [0] * len(self.pokecenter_ids)
        for i in self.visited_pokecenter_list:
            result[i] = 1
        return result
    
    def get_hm_move_obs(self):
        # workaround for hm moves
        # hm_moves = [0x0f, 0x13, 0x39, 0x46, 0x94]
        # result = [0] * len(hm_moves)
        # all_moves = self.get_party_moves()
        # for i, hm_move in enumerate(hm_moves):
        #     if hm_move in all_moves:
        #         result[i] = 1
        #         continue
        # return result

        # cut and surf for now,
        # can use flute, have silph scope
        # pokemon mansion switch status
        # 1 more placeholder
        map_id = self.current_map_id - 1
        if map_id in [0xA5, 0xD6, 0xD7, 0xD8]:
            pokemon_mansion_switch = self.read_bit(0xD796, 0)
        else:
            pokemon_mansion_switch = 0
        hm_moves = [self.can_use_cut, self.can_use_surf, 0, 0, 0, self.can_use_flute, self.have_silph_scope, pokemon_mansion_switch, 0]
        return hm_moves
    
    def get_hm_obs(self):
        hm_ids = [0xC4, 0xC5, 0xC6, 0xC7, 0xC8]
        items = self.get_items_in_bag()
        result = [0] * len(hm_ids)
        for i, hm_id in enumerate(hm_ids):
            if hm_id in items:
                result[i] = 1
                continue
        return result
    
    def get_items_obs(self):
        # items from self.get_items_in_bag()
        # add 0s to make it 20 items
        items = self.get_items_in_bag(one_indexed=1)
        items.extend([0] * (20 - len(items)))
        return items

    def get_items_quantity_obs(self):
        # items from self.get_items_quantity_in_bag()
        # add 0s to make it 20 items
        items = self.get_items_quantity_in_bag()
        items = self.scaled_encoding(items, 20)
        items.extend([0] * (20 - len(items)))
        return np.array(items, dtype=np.float32).reshape(-1, 1)

    def get_bag_full_obs(self):
        # D31D
        return [1 if self.read_m(0xD31D) >= 20 else 0]
    
    def get_last_10_map_ids_obs(self):
        return np.array(self.last_10_map_ids[:, 0], dtype=np.uint8)
    
    def get_last_10_map_step_since_obs(self):
        step_gotten = self.last_10_map_ids[:, 1]
        step_since = self.step_count - step_gotten
        return self.scaled_encoding(step_since, 5000).reshape(-1, 1)
    
    def get_last_10_coords_obs(self):
        # 10, 2
        # scale x with 45, y with 72
        result = []
        for coord in self.last_10_coords:
            result.append(min(coord[0] / 45, 1))
            result.append(min(coord[1] / 72, 1))
        return result
    
    def get_pokemon_ids_obs(self):
        return self.read_party(one_indexed=1)

    # def get_opp_pokemon_ids_obs(self):
    #     opp_pkmns = [self.read_m(addr) for addr in [0xD89D, 0xD89E, 0xD89F, 0xD8A0, 0xD8A1, 0xD8A2]]
    #     return [p + 1 if p != 0xff and p != 0 else 0 for p in opp_pkmns]
    
    def get_battle_pokemon_ids_obs(self):
        battle_pkmns = [self.read_m(addr) for addr in [0xcfe5, 0xd014]]
        return [p + 1 if p != 0xff and p != 0 else 0 for p in battle_pkmns]
    
    def get_party_types_obs(self):
        # 6 pokemon, 2 types each
        # start from D170 type1, D171 type2
        # next pokemon will be + 44
        # 0xff is no pokemon
        result = []
        for i in range(0, 44*6, 44):
            # 2 types per pokemon
            type1 = self.read_m(0xD170 + i)
            type2 = self.read_m(0xD171 + i)
            result.append(type1)
            result.append(type2)
        return [p + 1 if p != 0xff and p != 0 else 0 for p in result]
    
    def get_opp_types_obs(self):
        # 6 pokemon, 2 types each
        # start from D8A9 type1, D8AA type2
        # next pokemon will be + 44
        # 0xff is no pokemon
        result = []
        for i in range(0, 44*6, 44):
            # 2 types per pokemon
            type1 = self.read_m(0xD8A9 + i)
            type2 = self.read_m(0xD8AA + i)
            result.append(type1)
            result.append(type2)
        return [p + 1 if p != 0xff and p != 0 else 0 for p in result]
    
    def get_battle_types_obs(self):
        # CFEA type1, CFEB type2
        # d019 type1, d01a type2
        result = [self.read_m(0xcfea), self.read_m(0xCFEB), self.read_m(0xD019), self.read_m(0xD01A)]
        return [p + 1 if p != 0xff and p != 0 else 0 for p in result]
    
    def get_party_move_ids_obs(self):
        # D173 move1, D174 move2...
        # next pokemon will be + 44
        result = []
        for i in range(0, 44*6, 44):
            # 4 moves per pokemon
            moves = [self.read_m(addr + i) for addr in [0xD173, 0xD174, 0xD175, 0xD176]]
            result.extend(moves)
        return [p + 1 if p != 0xff and p != 0 else 0 for p in result]
    
    def get_opp_move_ids_obs(self):
        # D8AC move1, D8AD move2...
        # next pokemon will be + 44
        result = []
        for i in range(0, 44*6, 44):
            # 4 moves per pokemon
            moves = [self.read_m(addr + i) for addr in [0xD8AC, 0xD8AD, 0xD8AE, 0xD8AF]]
            result.extend(moves)
        return [p + 1 if p != 0xff and p != 0 else 0 for p in result]
    
    def get_battle_move_ids_obs(self):
        # CFED move1, CFEE move2
        # second pokemon starts from D003
        result = []
        for addr in [0xCFED, 0xD003]:
            moves = [self.read_m(addr + i) for i in range(4)]
            result.extend(moves)
        return [p + 1 if p != 0xff and p != 0 else 0 for p in result]
    
    def get_party_move_pps_obs(self):
        # D188 pp1, D189 pp2...
        # next pokemon will be + 44
        result = []
        for i in range(0, 44*6, 44):
            # 4 moves per pokemon
            pps = [self.read_m(addr + i) for addr in [0xD188, 0xD189, 0xD18A, 0xD18B]]
            result.extend(pps)
        return result
    
    def get_opp_move_pps_obs(self):
        # D8C1 pp1, D8C2 pp2...
        # next pokemon will be + 44
        result = []
        for i in range(0, 44*6, 44):
            # 4 moves per pokemon
            pps = [self.read_m(addr + i) for addr in [0xD8C1, 0xD8C2, 0xD8C3, 0xD8C4]]
            result.extend(pps)
        return result
    
    def get_battle_move_pps_obs(self):
        # CFFE pp1, CFFF pp2
        # second pokemon starts from D02D
        result = []
        for addr in [0xCFFE, 0xD02D]:
            pps = [self.read_m(addr + i) for i in range(4)]
            result.extend(pps)
        return result
    
    # def get_all_move_pps_obs(self):
    #     result = []
    #     result.extend(self.get_party_move_pps_obs())
    #     result.extend(self.get_opp_move_pps_obs())
    #     result.extend(self.get_battle_move_pps_obs())
    #     result = np.array(result, dtype=np.float32) / 30
    #     # every elemenet max is 1
    #     result = np.clip(result, 0, 1)
    #     return result
    
    def get_party_level_obs(self):
        # D18C level
        # next pokemon will be + 44
        result = []
        for i in range(0, 44*6, 44):
            level = self.read_m(0xD18C + i)
            result.append(level)
        return result
    
    def get_opp_level_obs(self):
        # D8C5 level
        # next pokemon will be + 44
        result = []
        for i in range(0, 44*6, 44):
            level = self.read_m(0xD8C5 + i)
            result.append(level)
        return result
    
    def get_battle_level_obs(self):
        # CFF3 level
        # second pokemon starts from D037
        result = []
        for addr in [0xCFF3, 0xD022]:
            level = self.read_m(addr)
            result.append(level)
        return result
    
    def get_all_level_obs(self):
        result = []
        result.extend(self.get_party_level_obs())
        result.extend(self.get_opp_level_obs())
        result.extend(self.get_battle_level_obs())
        result = np.array(result, dtype=np.float32) / 100
        # every elemenet max is 1
        result = np.clip(result, 0, 1)
        return result
    
    def get_party_hp_obs(self):
        # D16C hp
        # next pokemon will be + 44
        result = []
        for i in range(0, 44*6, 44):
            hp = self.read_hp(0xD16C + i)
            max_hp = self.read_hp(0xD18D + i)
            result.extend([hp, max_hp])
        return result

    def get_opp_hp_obs(self):
        # D8A5 hp
        # next pokemon will be + 44
        result = []
        for i in range(0, 44*6, 44):
            hp = self.read_hp(0xD8A5 + i)
            max_hp = self.read_hp(0xD8C6 + i)
            result.extend([hp, max_hp])
        return result
    
    def get_battle_hp_obs(self):
        # CFE6 hp
        # second pokemon starts from CFFC
        result = []
        for addr in [0xCFE6, 0xCFF4, 0xCFFC, 0xD00A]:
            hp = self.read_hp(addr)
            result.append(hp)
        return result
    
    def get_all_hp_obs(self):
        result = []
        result.extend(self.get_party_hp_obs())
        result.extend(self.get_opp_hp_obs())
        result.extend(self.get_battle_hp_obs())
        result = np.array(result, dtype=np.float32)
        # every elemenet max is 1
        result = np.clip(result, 0, 600) / 600
        return result
    
    def get_all_hp_pct_obs(self):
        hps = []
        hps.extend(self.get_party_hp_obs())
        hps.extend(self.get_opp_hp_obs())
        hps.extend(self.get_battle_hp_obs())
        # divide every hp by max hp
        hps = np.array(hps, dtype=np.float32)
        hps = hps.reshape(-1, 2)
        hps = hps[:, 0] / (hps[:, 1] + 0.00001)
        # every elemenet max is 1
        return hps
    
    def get_all_pokemon_dead_obs(self):
        # 1 if dead, 0 if alive
        hp_pct = self.get_all_hp_pct_obs()
        return [1 if hp <= 0 else 0 for hp in hp_pct]
    
    def get_battle_status_obs(self):
        # D057
        # 0 not in battle return 0, 0
        # 1 wild battle return 1, 0
        # 2 trainer battle return 0, 1
        # -1 lost battle return 0, 0
        result = []
        status = self.battle_type
        if status == 1:
            result = [1, 0]
        elif status == 2:
            result = [0, 1]
        else:
            result = [0, 0]
        return result
    
    # def get_reward_check_obs(self):
    #     reward_steps = [2500, 5000, 7500, 10000]
    #     result = []
    #     for step in reward_steps:
    #         if self.step_count > step:
    #             result.append(1 if self.past_rewards[step-1] - self.past_rewards[0] < 1 else 0)
    #         else:
    #             result.append(0)
    #     return result

    # def get_vector_raw_obs(self):
    #     obs = []
    #     obs.extend(self.get_badges_obs())
    #     obs.extend(self.get_money_obs())
    #     obs.extend(self.get_last_pokecenter_obs())
    #     obs.extend(self.get_visited_pokecenter_obs())
    #     obs.extend(self.get_hm_move_obs())
    #     obs.extend(self.get_hm_obs())
    #     # obs.extend(self.get_items_obs())
    #     obs.extend(self.get_items_quantity_obs())
    #     obs.extend(self.get_bag_full_obs())
    #     # obs.extend(self.get_last_10_map_ids_obs())
    #     obs.extend(self.get_last_10_coords_obs())

    #     obs.extend(self.get_all_move_pps_obs())
    #     obs.extend(self.get_all_level_obs())
    #     obs.extend(self.get_all_hp_obs())
    #     obs.extend(self.get_all_hp_pct_obs())
    #     obs.extend(self.get_all_pokemon_dead_obs())
    #     obs.extend(self.get_battle_status_obs())
    #     # obs.extend(self.get_swap_pokemon_obs())
    #     obs.extend(self.get_reward_check_obs())
    #     obs = np.array(obs, dtype=np.float32)
    #     obs = np.clip(obs, 0, 1)
    #     obs = obs * 255
    #     # check if there is any invalid value
    #     # print(f'invalid value: {np.isnan(obs).any()}')    
    #     obs = obs.astype(np.uint8)
    #     return obs
    
    def fix_pokemon_type(self, ptype: int) -> int:
        if ptype < 9:
            return ptype
        elif ptype < 27:
            return ptype - 11
        else:
            print(f'invalid pokemon type: {ptype}')
            return -1
        
    def get_pokemon_types(self, start_addr):
        return [self.fix_pokemon_type(self.read_m(start_addr + i)) + 1 for i in range(2)]
        
    def get_all_pokemon_types_obs(self):
        # 6 party pokemon types start from D170
        # 6 enemy pokemon types start from D8A9
        party_type_addr = 0xD170
        enemy_type_addr = 0xD8A9
        result = []
        pokemon_count = self.read_num_poke()
        if pokemon_count > 6:
            print(f'invalid pokemon count: {pokemon_count}')
            pokemon_count = 6
            self.debug_save()
        for i in range(pokemon_count):
            # 2 types per pokemon
            ptypes = self.get_pokemon_types(party_type_addr + i * 44)
            result.append(ptypes)
        remaining_pokemon = 6 - pokemon_count
        for i in range(remaining_pokemon):
            result.append([0, 0])
        if self.is_in_battle():
            # zero padding if not in battle, reduce dimension
            if not self.is_wild_battle():
                pokemon_count = self.read_opp_pokemon_num()
                if pokemon_count > 6:
                    print(f'invalid opp_pokemon count: {pokemon_count}')
                    pokemon_count = 6
                    self.debug_save()
                for i in range(pokemon_count):
                    # 2 types per pokemon
                    ptypes = self.get_pokemon_types(enemy_type_addr + i * 44)
                    result.append(ptypes)
                remaining_pokemon = 6 - pokemon_count
                for i in range(remaining_pokemon):
                    result.append([0, 0])
            else:
                wild_ptypes = self.get_pokemon_types(0xCFEA)  # 2 ptypes only, add padding for remaining 5
                result.append(wild_ptypes)
                result.extend([[0, 0]] * 5)
        else:
            result.extend([[0, 0]] * 6)
        result = np.array(result, dtype=np.uint8)  # shape (24,)
        assert result.shape == (12, 2), f'invalid ptypes shape: {result.shape}'  # set PYTHONOPTIMIZE=1 to disable assert
        return result
    
    def get_pokemon_status(self, addr):
        # status
        # bit 0 - 6
        # one byte has 8 bits, bit unused: 7
        statuses = [self.read_bit(addr, i) for i in range(7)]
        return statuses  # shape (7,)
    
    def get_one_pokemon_obs(self, start_addr, team, position, is_wild=False):
        # team 0 = my team, 1 = opp team
        # 1 pokemon, address start from start_addr
        # +0 = id
        # +5 = type1 (15 types) (physical 0 to 8 and special 20 to 26)  + 1 to be 1 indexed, 0 is no pokemon/padding
        # +6 = type2 (15 types)
        # +33 = level
        # +4 = status (bit 0-6)
        # +1 = current hp (2 bytes)
        # +34 = max hp (2 bytes)
        # +36 = attack (2 bytes)
        # +38 = defense (2 bytes)
        # +40 = speed (2 bytes)
        # +42 = special (2 bytes)
        # exclude id, type1, type2
        result = []
        # status
        status = self.get_pokemon_status(start_addr + 4)
        result.extend(status)
        # level
        level = self.scaled_encoding(self.read_m(start_addr + 33), 100)
        result.append(level)
        # hp
        hp = self.scaled_encoding(self.read_double(start_addr + 1), 250)
        result.append(hp)
        # max hp
        max_hp = self.scaled_encoding(self.read_double(start_addr + 34), 250)
        result.append(max_hp)
        # attack
        attack = self.scaled_encoding(self.read_double(start_addr + 36), 134)
        result.append(attack)
        # defense
        defense = self.scaled_encoding(self.read_double(start_addr + 38), 180)
        result.append(defense)
        # speed
        speed = self.scaled_encoding(self.read_double(start_addr + 40), 140)
        result.append(speed)
        # special
        special = self.scaled_encoding(self.read_double(start_addr + 42), 154)
        result.append(special)
        # is alive
        is_alive = 1 if hp > 0 else 0
        result.append(is_alive)
        # is in battle, check position 0 indexed against the following addr
        if is_wild:
            in_battle = 1
        else:
            if self.is_in_battle():
                if team == 0:
                    in_battle = 1 if position == self.read_m(0xCC35) else 0
                else:
                    in_battle = 1 if position == self.read_m(0xCFE8) else 0
            else:
                in_battle = 0
        result.append(in_battle)
        # my team 0 / opp team 1
        result.append(team)
        # position 0 to 5, one hot, 5 elements, first pokemon is all 0
        result.extend(self.one_hot_encoding(position, 5))
        # is swapping this pokemon
        if team == 0:
            swap_mon_pos = self.read_swap_mon_pos()
            if swap_mon_pos != -1:
                is_swapping = 1 if position == swap_mon_pos else 0
            else:
                is_swapping = 0
        else:
            is_swapping = 0
        result.append(is_swapping)
        return result

    def get_party_pokemon_obs(self):
        # 6 party pokemons start from D16B
        # 2d array, 6 pokemons, N features
        result = np.zeros((6, self.n_pokemon_features), dtype=np.float32)
        pokemon_count = self.read_num_poke()
        for i in range(pokemon_count):
            result[i] = self.get_one_pokemon_obs(0xD16B + i * 44, 0, i)
        for i in range(pokemon_count, 6):
            result[i] = np.zeros(self.n_pokemon_features, dtype=np.float32)
        return result

    def read_opp_pokemon_num(self):
        return self.read_m(0xD89C)
    
    def get_battle_base_pokemon_obs(self, start_addr, team, position=0):
        # CFE5
        result = []
        # status
        status = self.get_pokemon_status(start_addr + 4)
        result.extend(status)
        # level
        level = self.scaled_encoding(self.read_m(start_addr + 14), 100)
        result.append(level)
        # hp
        hp = self.scaled_encoding(self.read_double(start_addr + 1), 250)
        result.append(hp)
        # max hp
        max_hp = self.scaled_encoding(self.read_double(start_addr + 15), 250)
        result.append(max_hp)
        # attack
        attack = self.scaled_encoding(self.read_double(start_addr + 17), 134)
        result.append(attack)
        # defense
        defense = self.scaled_encoding(self.read_double(start_addr + 19), 180)
        result.append(defense)
        # speed
        speed = self.scaled_encoding(self.read_double(start_addr + 21), 140)
        result.append(speed)
        # special
        special = self.scaled_encoding(self.read_double(start_addr + 23), 154)
        result.append(special)
        # is alive
        is_alive = 1 if hp > 0 else 0
        result.append(is_alive)
        # is in battle, check position 0 indexed against the following addr
        in_battle = 1
        result.append(in_battle)
        # my team 0 / opp team 1
        result.append(team)
        # position 0 to 5, one hot, 5 elements, first pokemon is all 0
        result.extend(self.one_hot_encoding(position, 5))
        is_swapping = 0
        result.append(is_swapping)
        return result
    
    def get_wild_pokemon_obs(self):
        start_addr = 0xCFE5
        return self.get_battle_base_pokemon_obs(start_addr, team=1)

    def get_opp_pokemon_obs(self):
        # 6 enemy pokemons start from D8A4
        # 2d array, 6 pokemons, N features
        result = []
        if self.is_in_battle():
            if not self.is_wild_battle():
                pokemon_count = self.read_opp_pokemon_num()
                for i in range(pokemon_count):
                    if i == self.read_m(0xCFE8):
                        # in battle
                        result.append(self.get_battle_base_pokemon_obs(0xCFE5, 1, i))
                    else:
                        result.append(self.get_one_pokemon_obs(0xD8A4 + i * 44, 1, i))
                remaining_pokemon = 6 - pokemon_count
                for i in range(remaining_pokemon):
                    result.append([0] * self.n_pokemon_features)
            else:
                # wild battle, take the battle pokemon
                result.append(self.get_wild_pokemon_obs())
                for i in range(5):
                    result.append([0] * self.n_pokemon_features)
        else:
            return np.zeros((6, self.n_pokemon_features), dtype=np.float32)
        result = np.array(result, dtype=np.float32)
        return result
    
    def get_all_pokemon_obs(self):
        # 6 party pokemons start from D16B
        # 6 enemy pokemons start from D8A4
        # gap between each pokemon is 44
        party = self.get_party_pokemon_obs()
        opp = self.get_opp_pokemon_obs()
        # print(f'party shape: {party.shape}, opp shape: {opp.shape}')
        result = np.concatenate([party, opp], axis=0)
        return result  # shape (12, 22)
    
    def get_party_pokemon_ids_obs(self):
        # 6 party pokemons start from D16B
        # 1d array, 6 pokemons, 1 id
        result = []
        pokemon_count = self.read_num_poke()
        for i in range(pokemon_count):
            result.append(self.read_m(0xD16B + i * 44) + 1)
        remaining_pokemon = 6 - pokemon_count
        for i in range(remaining_pokemon):
            result.append(0)
        result = np.array(result, dtype=np.uint8)
        return result
    
    def get_opp_pokemon_ids_obs(self):
        # 6 enemy pokemons start from D8A4
        # 1d array, 6 pokemons, 1 id
        result = []
        if self.is_in_battle():
            if not self.is_wild_battle():
                pokemon_count = self.read_opp_pokemon_num()
                for i in range(pokemon_count):
                    result.append(self.read_m(0xD8A4 + i * 44) + 1)
                remaining_pokemon = 6 - pokemon_count
                for i in range(remaining_pokemon):
                    result.append(0)
            else:
                # wild battle, take the battle pokemon
                result.append(self.read_m(0xCFE5) + 1)
                for i in range(5):
                    result.append(0)
        else:
            return np.zeros(6, dtype=np.uint8)
        result = np.array(result, dtype=np.uint8)
        return result
    
    def get_all_pokemon_ids_obs(self):
        # 6 party pokemons start from D16B
        # 6 enemy pokemons start from D8A4
        # gap between each pokemon is 44
        party = self.get_party_pokemon_ids_obs()
        opp = self.get_opp_pokemon_ids_obs()
        result = np.concatenate((party, opp), axis=0)
        return result
    
    def get_one_pokemon_move_ids_obs(self, start_addr):
        # 4 moves
        return [self.read_m(start_addr + i) for i in range(4)]
    
    def get_party_pokemon_move_ids_obs(self):
        # 6 party pokemons start from D173
        # 2d array, 6 pokemons, 4 moves
        result = []
        pokemon_count = self.read_num_poke()
        for i in range(pokemon_count):
            result.append(self.get_one_pokemon_move_ids_obs(0xD173 + (i * 44)))
        remaining_pokemon = 6 - pokemon_count
        for i in range(remaining_pokemon):
            result.append([0] * 4)
        result = np.array(result, dtype=np.uint8)
        return result

    def get_opp_pokemon_move_ids_obs(self):
        # 6 enemy pokemons start from D8AC
        # 2d array, 6 pokemons, 4 moves
        result = []
        if self.is_in_battle():
            if not self.is_wild_battle():
                pokemon_count = self.read_opp_pokemon_num()
                for i in range(pokemon_count):
                    result.append(self.get_one_pokemon_move_ids_obs(0xD8AC + (i * 44)))
                remaining_pokemon = 6 - pokemon_count
                for i in range(remaining_pokemon):
                    result.append([0] * 4)
            else:
                # wild battle, take the battle pokemon
                result.append(self.get_one_pokemon_move_ids_obs(0xCFED))
                for i in range(5):
                    result.append([0] * 4)
        else:
            return np.zeros((6, 4), dtype=np.uint8)
        result = np.array(result, dtype=np.uint8)
        return result
    
    def get_all_move_ids_obs(self):
        # 6 party pokemons start from D173
        # 6 enemy pokemons start from D8AC
        # gap between each pokemon is 44
        party = self.get_party_pokemon_move_ids_obs()
        opp = self.get_opp_pokemon_move_ids_obs()
        result = np.concatenate((party, opp), axis=0)
        return result  # shape (12, 4)
    
    def get_one_pokemon_move_pps_obs(self, start_addr):
        # 4 moves
        result = np.zeros((4, 2), dtype=np.float32)
        for i in range(4):
            pp = self.scaled_encoding(self.read_m(start_addr + i), 30)
            have_pp = 1 if pp > 0 else 0
            result[i] = [pp, have_pp]
        return result
    
    def get_party_pokemon_move_pps_obs(self):
        # 6 party pokemons start from D188
        # 2d array, 6 pokemons, 8 features
        # features: pp, have pp
        result = np.zeros((6, 4, 2), dtype=np.float32)
        pokemon_count = self.read_num_poke()
        for i in range(pokemon_count):
            result[i] = self.get_one_pokemon_move_pps_obs(0xD188 + (i * 44))
        for i in range(pokemon_count, 6):
            result[i] = np.zeros((4, 2), dtype=np.float32)
        return result
    
    def get_opp_pokemon_move_pps_obs(self):
        # 6 enemy pokemons start from D8C1
        # 2d array, 6 pokemons, 8 features
        # features: pp, have pp
        result = np.zeros((6, 4, 2), dtype=np.float32)
        if self.is_in_battle():
            if not self.is_wild_battle():
                pokemon_count = self.read_opp_pokemon_num()
                for i in range(pokemon_count):
                    result[i] = self.get_one_pokemon_move_pps_obs(0xD8C1 + (i * 44))
                for i in range(pokemon_count, 6):
                    result[i] = np.zeros((4, 2), dtype=np.float32)
            else:
                # wild battle, take the battle pokemon
                result[0] = self.get_one_pokemon_move_pps_obs(0xCFFE)
                for i in range(1, 6):
                    result[i] = np.zeros((4, 2), dtype=np.float32)
        else:
            return np.zeros((6, 4, 2), dtype=np.float32)
        return result
    
    def get_all_move_pps_obs(self):
        # 6 party pokemons start from D188
        # 6 enemy pokemons start from D8C1
        party = self.get_party_pokemon_move_pps_obs()
        opp = self.get_opp_pokemon_move_pps_obs()
        result = np.concatenate((party, opp), axis=0)
        return result
    
    def get_all_item_ids_obs(self):
        # max 85
        return np.array(self.get_items_obs(), dtype=np.uint8)
    
    def get_all_event_ids_obs(self):
        # max 249
        # padding_idx = 0
        # change dtype to uint8 to save space
        return np.array(self.last_10_event_ids[:, 0] + 1, dtype=np.uint8)
    
    def get_all_event_step_since_obs(self):
        step_gotten = self.last_10_event_ids[:, 1]  # shape (10,)
        step_since = self.step_count - step_gotten
        # step_count - step_since and scaled_encoding
        return self.scaled_encoding(step_since, 10000).reshape(-1, 1)  # shape (10,)
    
    def get_last_coords_obs(self):
        # 2 elements
        coord = self.last_10_coords[0]
        max_x = 45
        max_y = 72
        cur_map_id = self.current_map_id - 1
        if cur_map_id in MAP_ID_REF:
            max_x = MAP_DICT[MAP_ID_REF[cur_map_id]]['width']
            max_y = MAP_DICT[MAP_ID_REF[cur_map_id]]['height']
            if max_x == 0:
                if cur_map_id not in [231]:  # 231 is expected
                    print(f'invalid max_x: {max_x}, map_id: {cur_map_id}')
                max_x = 45
            if max_y == 0:
                if cur_map_id not in [231]:
                    print(f'invalid max_y: {max_y}, map_id: {cur_map_id}')
                max_y = 72
        return [self.scaled_encoding(coord[0], max_x), self.scaled_encoding(coord[1], max_y)]
    
    def get_num_turn_in_battle_obs(self):
        if self.is_in_battle:
            return self.scaled_encoding(self.read_m(0xCCD5), 30)
        else:
            return 0
        
    def get_stage_obs(self):
        # set stage obs to 14 for now
        if not self.enable_stage_manager:
            return np.zeros(28, dtype=np.uint8)
        # self.stage_manager.n_stage_started : int
        # self.stage_manager.n_stage_ended : int
        # 28 elements, 14 n_stage_started, 14 n_stage_ended
        result = np.zeros(28, dtype=np.uint8)
        result[:self.stage_manager.n_stage_started] = 1
        result[14:14+self.stage_manager.n_stage_ended] = 1
        return result  # shape (28,)
    
    def get_all_raw_obs(self):
        obs = []
        obs.extend(self.get_badges_obs())
        obs.extend(self.get_money_obs())
        obs.extend(self.get_last_pokecenter_obs())
        obs.extend(self.get_visited_pokecenter_obs())
        obs.extend(self.get_hm_move_obs())
        obs.extend(self.get_hm_obs())
        obs.extend(self.get_battle_status_obs())
        pokemon_count = self.read_num_poke()
        obs.extend([self.scaled_encoding(pokemon_count, 6)])  # number of pokemon
        obs.extend([1 if pokemon_count == 6 else 0])  # party full
        obs.extend([self.scaled_encoding(self.read_m(0xD31D), 20)])  # bag num items
        obs.extend(self.get_bag_full_obs())  # bag full
        obs.extend(self.get_last_coords_obs())  # last coords x, y
        obs.extend([self.get_num_turn_in_battle_obs()])  # num turn in battle
        obs.extend(self.get_stage_obs())  # stage manager
        obs.extend(self.get_level_manager_obs())  # level manager
        obs.extend(self.get_is_box_mon_higher_level_obs())  # is box mon higher level
        # obs.extend(self.get_reward_check_obs())  # reward check
        return np.array(obs, dtype=np.float32)
    
    def get_level_manager_obs(self):
        # self.current_level by one hot encoding
        return self.one_hot_encoding(self.current_level, 10)
    
    @property
    def is_box_mon_higher_level(self):
        # check if num_mon_in_box is different than last_num_mon_in_box
        if self.last_num_mon_in_box == 0:
            return False
        
        if self.last_num_mon_in_box == self.num_mon_in_box:
            return self._is_box_mon_higher_level
        
        self._is_box_mon_higher_level = False
        # check if there is any pokemon in box with higher level than the lowest level pokemon in party
        party_count = self.read_num_poke()
        if party_count < 6:
            return False
        party_addr_start = 0xD16B
        box_mon_addr_start = 0xda96
        num_mon_in_box = self.read_m(0xda80)  # wBoxCount  wNumInBox
        party_levels = [self.read_m(party_addr_start + i * 44 + 33) for i in range(party_count)]
        lowest_party_level = min(party_levels)
        box_levels = [self.read_m(box_mon_addr_start + i * 33 + 3) for i in range(num_mon_in_box)]
        highest_box_level = max(box_levels) if box_levels else 0
        if highest_box_level > lowest_party_level:
            self._is_box_mon_higher_level = True
        # self.last_num_mon_in_box = self.num_mon_in_box  # this is updated in step()
        return self._is_box_mon_higher_level
    
    def get_is_box_mon_higher_level_obs(self):
        return np.array([self.is_box_mon_higher_level], dtype=np.float32)

    def get_last_map_id_obs(self):
        return np.array([self.last_10_map_ids[0]], dtype=np.uint8)
    
    @property
    def current_map_id(self):
        return self.last_10_map_ids[0, 0]
    
    @property
    def current_coords(self):
        return self.last_10_coords[0]
    
    def get_in_battle_mask_obs(self):
        return np.array([self.is_in_battle()], dtype=np.float32)

    def scripted_manage_items(self):
        items = self.get_items_in_bag()
        if self._last_item_count == len(items):
            return
        
        if self.is_in_battle() or self.read_m(0xFFB0) == 0:  # hWY in menu
            return
        
        # pokeballs = [0x01, 0x02, 0x03, 0x04]

        if len(items) == 20:
            # bag full, delete 1 item
            # do not delete pokeballs and ALL_KEY_ITEMS
            # try to delete the last item first
            # if it is not the last item, swap with the last item
            # set the address after the last item to 255
            # set the address after the last quantity to 0
            tmp_item = items[-1]
            tmp_item_quantity = self.get_items_quantity_in_bag()[-1]
            
            deleted = False
            for i in range(19, -1, -1):
                if items[i] not in ALL_GOOD_ITEMS:
                    if i == 19:
                        # delete the last item
                        self.pyboy.set_memory_value(0xD31E + i*2, 0xff)
                        self.pyboy.set_memory_value(0xD31F + i*2, 0)
                    else:
                        # swap with last item
                        self.pyboy.set_memory_value(0xD31E + i*2, tmp_item)
                        self.pyboy.set_memory_value(0xD31F + i*2, tmp_item_quantity)
                        # set last item to 255
                        self.pyboy.set_memory_value(0xD31E + 19*2, 0xff)
                        self.pyboy.set_memory_value(0xD31F + 19*2, 0)
                    # print(f'Delete item: {items[i]}')
                    deleted = True
                    break

            if not deleted:
                # print(f'Warning: no item deleted, bag full, delete good items instead')
                # delete good items if no other items
                # from first to last good items priority
                for good_item in GOOD_ITEMS_PRIORITY:
                    if good_item in items:
                        idx = items.index(good_item)
                        if idx == 19:
                            # delete the last item
                            self.pyboy.set_memory_value(0xD31E + idx*2, 0xff)
                            self.pyboy.set_memory_value(0xD31F + idx*2, 0)
                        else:
                            # swap with last item
                            self.pyboy.set_memory_value(0xD31E + idx*2, tmp_item)
                            self.pyboy.set_memory_value(0xD31F + idx*2, tmp_item_quantity)
                            # set last item to 255
                            self.pyboy.set_memory_value(0xD31E + 19*2, 0xff)
                            self.pyboy.set_memory_value(0xD31F + 19*2, 0)
                        # print(f'Delete item: {items[idx]}')
                        deleted = True
                        break

            # reset cache and get items again
            self._items_in_bag = None
            items = self.get_items_in_bag()
            self.pyboy.set_memory_value(0xD31D, len(items))
        
        item_idx_ptr = 0
        # sort good items to the front based on priority
        for good_item in GOOD_ITEMS_PRIORITY:
            if good_item in items:
                all_items_quantity = self.get_items_quantity_in_bag()
                idx = items.index(good_item)
                if idx == item_idx_ptr:
                    # already in the correct position
                    item_idx_ptr += 1
                    continue
                cur_item_quantity = all_items_quantity[idx]
                tmp_item = items[item_idx_ptr]
                tmp_item_quantity = all_items_quantity[item_idx_ptr]
                # print(f'Swapping item {item_idx_ptr}:{tmp_item}/{tmp_item_quantity} with {idx}:{good_item}/{cur_item_quantity}')
                # swap
                self.pyboy.set_memory_value(0xD31E + item_idx_ptr*2, good_item)
                self.pyboy.set_memory_value(0xD31F + item_idx_ptr*2, cur_item_quantity)
                self.pyboy.set_memory_value(0xD31E + idx*2, tmp_item)
                self.pyboy.set_memory_value(0xD31F + idx*2, tmp_item_quantity)
                item_idx_ptr += 1
                # reset cache and get items again
                self._items_in_bag = None
                items = self.get_items_in_bag()
                # print(f'Moved good item: {good_item} to pos: {item_idx_ptr}')
        self._last_item_count = len(self.get_items_in_bag())

    def get_best_item_from_list(self, target_item_list, given_item_list):
        # return the best item in target_item_list that is in given_item_list
        # if no item in target_item_list is in given_item_list, return None
        # target_item_list and given_item_list are list of item id
        for item in target_item_list:
            if item in given_item_list:
                return item, target_item_list.index(item)
        return None, None
    
    def get_mart_items(self):
        map_id = int(self.current_map_id - 1)
        x, y = self.current_coords
        # key format map_id@x,y
        dict_key = f'{map_id}@{x},{y}'
        if dict_key in MART_ITEMS_ID_DICT:
            mart_matched = MART_ITEMS_ID_DICT[dict_key]
            # match direction in mart_matched['dir']
            facing_direction = self.read_m(0xC109)  # wSpritePlayerStateData1FacingDirection
            direction = None
            if facing_direction == 0:  # down
                direction = 'down'
            elif facing_direction == 4:  # up
                direction = 'up'
            elif facing_direction == 8:  # left
                direction = 'left'
            elif facing_direction == 12:  # right
                direction = 'right'
            if direction is None:
                print(f'Warning: invalid facing direction: {facing_direction}')
                return None
            if direction == mart_matched['dir']:
                return mart_matched['items']    
        return None

    def get_item_price_by_id(self, item_id):
        # must have or error out
        return ITEM_TM_IDS_PRICES[item_id]
    
    def write_bcd(self, num):
        return ((num // 10) << 4) + (num % 10)
    
    def add_money(self, amount):
        if not amount:
            return
        money = self.read_money()
        # read_money() function
        # return (100 * 100 * self.read_bcd(self.read_m(0xD347)) + 
        #         100 * self.read_bcd(self.read_m(0xD348)) +
        #         self.read_bcd(self.read_m(0xD349)))
        # def read_bcd(self, num):
        #     return 10 * ((num >> 4) & 0x0f) + (num & 0x0f)
        if amount < 0:
            # deduct money
            money = max(0, money + amount)
        else:
            money += amount
        money = min(money, 999999)
        # self.pyboy.set_memory_value
        # it is in bcd format so need to convert to bcd
        self.pyboy.set_memory_value(0xD347, self.write_bcd(money // 10000))
        self.pyboy.set_memory_value(0xD348, self.write_bcd((money % 10000) // 100))
        self.pyboy.set_memory_value(0xD349, self.write_bcd(money % 100))

    def sell_or_delete_item(self, is_sell, good_item_id=None):
        # bag full, delete 1 item
        # do not delete pokeballs and ALL_KEY_ITEMS
        # try to delete the last item first
        # if it is not the last item, swap with the last item
        # set the address after the last item to 255
        # set the address after the last quantity to 0
        
        items = self.get_items_in_bag()
        while len(items) > 0:
            items_quantity = self.get_items_quantity_in_bag()
            # if len(items) == 0:
            #     break
            tmp_item = items[-1]
            tmp_item_quantity = items_quantity[-1]
            deleted = None
            for i in range(len(items) - 1, -1, -1):
                if items[i] not in ALL_GOOD_ITEMS:
                    if i == 19:
                        # delete the last item
                        self.pyboy.set_memory_value(0xD31E + i*2, 0xff)
                        self.pyboy.set_memory_value(0xD31F + i*2, 0)
                    else:
                        # swap with last item
                        self.pyboy.set_memory_value(0xD31E + i*2, tmp_item)
                        self.pyboy.set_memory_value(0xD31F + i*2, tmp_item_quantity)
                        # set last item to 255
                        self.pyboy.set_memory_value(0xD31E + (len(items) - 1)*2, 0xff)
                        self.pyboy.set_memory_value(0xD31F + (len(items) - 1)*2, 0)
                    # print(f'Delete item: {items[i]}')
                    deleted = items[i]
                    if is_sell:
                        self.add_money(self.get_item_price_by_id(deleted) // 2 * items_quantity[i])
                    # reset cache and get items again
                    self._items_in_bag = None
                    items = self.get_items_in_bag()
                    self.pyboy.set_memory_value(0xD31D, len(items))
                    # items_quantity = self.get_items_quantity_in_bag()
                    break
            if deleted is None:
                # no more item to delete
                break

        if good_item_id is not None:
            items = self.get_items_in_bag()
            if good_item_id in items:
                items_quantity = self.get_items_quantity_in_bag()
                tmp_item = items[-1]
                tmp_item_quantity = items_quantity[-1]
                idx = items.index(good_item_id)
                if idx == 19:
                    # delete the last item
                    self.pyboy.set_memory_value(0xD31E + idx*2, 0xff)
                    self.pyboy.set_memory_value(0xD31F + idx*2, 0)
                else:
                    # swap with last item
                    self.pyboy.set_memory_value(0xD31E + idx*2, tmp_item)
                    self.pyboy.set_memory_value(0xD31F + idx*2, tmp_item_quantity)
                    # set last item to 255
                    self.pyboy.set_memory_value(0xD31E + 19*2, 0xff)
                    self.pyboy.set_memory_value(0xD31F + 19*2, 0)
                # print(f'Delete item: {items[idx]}')
                deleted = good_item_id
                if is_sell:
                    self.add_money(self.get_item_price_by_id(deleted) // 2 * items_quantity[idx])
                # reset cache and get items again
                self._items_in_bag = None
                items = self.get_items_in_bag()
                self.pyboy.set_memory_value(0xD31D, len(items))
                # items_quantity = self.get_items_quantity_in_bag()

    def buy_item(self, item_id, quantity, price):
        # add the item at the end of the bag
        # deduct money by price * quantity
        bag_items = self.get_items_in_bag()
        bag_items_quantity = self.get_items_quantity_in_bag()
        if len(bag_items) >= 20:
            # bag full
            return
        if item_id in bag_items:
            idx = bag_items.index(item_id)
            bag_items_quantity[idx] += quantity
            self.pyboy.set_memory_value(0xD31F + idx*2, bag_items_quantity[idx])
        else:
            idx = len(bag_items)
            self.pyboy.set_memory_value(0xD31E + idx*2, item_id)
            self.pyboy.set_memory_value(0xD31F + idx*2, quantity)
            # check if this is the last item in bag
            if idx != 19:
                # if not then need to set the next item id to 255
                self.pyboy.set_memory_value(0xD31E + idx*2 + 2, 0xff)
                self.pyboy.set_memory_value(0xD31F + idx*2 + 2, 0)
        self.add_money(-price * quantity)
        # reset cache and get items again
        self._items_in_bag = None

    def scripted_buy_items(self):
        if self.read_m(0xFFB0) == 0:  # hWY in menu
            return False
        # check mart items
        # if mart has items in GOOD_ITEMS_PRIORITY list (the last item is the highest priority)
        #  check if have enough (10) of the item
        #   if not enough, check if have enough money to buy the item
        #    if have enough money, check if bag is 19/20
        #     if bag is 19/20, sell 1 item
        #      handle if all items are key or good items
        #     buy the item by deducting money and adding item to bag

        # will try to buy 10 best pokeballs offered by mart
        # and 10 best potions and revives offered by mart
        mart_items = self.get_mart_items()
        if not mart_items:
            # not in mart or incorrect x, y
            # or mart_items is empty for purchasable items
            return False
        bag_items = self.get_items_in_bag()
        item_list_to_buy = [POKEBALL_PRIORITY, POTION_PRIORITY, REVIVE_PRIORITY]
        target_quantity = 10
        for n_list, item_list in enumerate(item_list_to_buy):
            if self.stage_manager.stage == 11:
                if n_list == 0:
                    # pokeball
                    target_quantity = 5
                elif n_list == 1:
                    # potion
                    target_quantity = 20
                elif n_list == 2:
                    # revive
                    target_quantity = 10
            best_in_mart_id, best_in_mart_priority = self.get_best_item_from_list(item_list, mart_items)
            best_in_bag_id, best_in_bag_priority = self.get_best_item_from_list(item_list, bag_items)
            best_in_bag_idx = bag_items.index(best_in_bag_id) if best_in_bag_id is not None else None
            best_in_bag_quantity = self.get_items_quantity_in_bag()[best_in_bag_idx] if best_in_bag_idx is not None else None
            if best_in_mart_id is None:
                continue
            if best_in_bag_priority is not None:
                if n_list == 0 and best_in_mart_priority - best_in_bag_priority > 1:
                    # having much better pokeball in bag, skip buying
                    continue
                elif n_list == 1 and best_in_mart_priority - best_in_bag_priority > 2:
                    # having much better potion in bag, skip buying
                    continue
                # revive only have 2 types so ok to buy if insufficient
                if best_in_bag_id is not None and best_in_bag_priority < best_in_mart_priority and best_in_bag_quantity >= target_quantity:
                    # already have better item in bag with desired quantity
                    continue
                if best_in_bag_quantity is not None and best_in_bag_priority == best_in_mart_priority and best_in_bag_quantity >= target_quantity:
                    # same item
                    # and already have enough
                    continue
            item_price = self.get_item_price_by_id(best_in_mart_id)
            # try to sell items
            if best_in_bag_priority is not None and best_in_bag_priority > best_in_mart_priority:
                # having worse item in bag, sell it
                if n_list == 0 and best_in_bag_priority - best_in_mart_priority > 1:
                    # having much worse pokeball in bag
                    self.sell_or_delete_item(is_sell=True, good_item_id=best_in_bag_id)
                elif n_list == 1 and best_in_bag_priority - best_in_mart_priority > 2:
                    # having much worse potion in bag
                    self.sell_or_delete_item(is_sell=True, good_item_id=best_in_bag_id)
            else:
                self.sell_or_delete_item(is_sell=True)
            # get items again
            bag_items = self.get_items_in_bag()
            if best_in_mart_id not in bag_items and len(bag_items) >= 19:
                # is new item and bag is full
                # bag is full even after selling
                break
            if self.read_money() < item_price:
                # not enough money
                continue
            if best_in_bag_quantity is None:
                needed_quantity = target_quantity
            elif best_in_bag_priority == best_in_mart_priority:
                # item in bag is same
                needed_quantity = target_quantity - best_in_bag_quantity
            elif best_in_bag_priority > best_in_mart_priority:
                # item in bag is worse
                needed_quantity = target_quantity
            elif best_in_bag_priority < best_in_mart_priority:
                # item in bag is better, but not enough quantity
                if best_in_mart_id in bag_items:
                    mart_item_in_bag_idx = bag_items.index(best_in_mart_id)
                    needed_quantity = target_quantity - self.get_items_quantity_in_bag()[mart_item_in_bag_idx] - best_in_bag_quantity
                else:
                    needed_quantity = target_quantity - best_in_bag_quantity
            if needed_quantity < 1:
                # already have enough
                continue
            affordable_quantity = min(needed_quantity, (self.read_money() // item_price))
            self.buy_item(best_in_mart_id, affordable_quantity, item_price)
            # reset cache and get items again
            self._items_in_bag = None
            bag_items = self.get_items_in_bag()
            self.pyboy.set_memory_value(0xD31D, len(bag_items))
            self.pyboy.set_memory_value(RAM.wBagSavedMenuItem.value, 0x0)
            # print(f'Bought item: {best_in_mart_id} x {affordable_quantity}')
        self.use_mart_count += 1
        # reset item count to trigger scripted_manage_items
        self._last_item_count = 0
        return True
    
    def minor_patch(self):
        cur_map_id = self.current_map_id - 1
        # to ensure have atleast in safari entrance 500 to prevent hard stuck
        if cur_map_id == 0x9C:
            current_money = self.read_money()
            if current_money < 501:
                self.add_money(500 - current_money)
        # to ensure safari has infinite steps
        if 0xE1 >= cur_map_id >= 0xD9:
            # if don't have hm surf and (gold teeth or hm strength), set safari steps to 1 (infinite)
            # else do not set safari steps, let it exits safari
            bag_items = self.get_items_in_bag()
            if 0xC6 in bag_items and (0x40 in bag_items or 0xC7 in bag_items):
                pass
            else:
                if self.read_ram_m(RAM.wSafariSteps) == 0:
                    self.pyboy.set_memory_value(RAM.wSafariSteps.value, 1)
        # nerf spinner tiles, make it spin 1 tile only
        if cur_map_id in [0x2D, 0xC7, 0xC8, 0xC9, 0xCA]:
            self.pyboy.set_memory_value(RAM.wSimulatedJoypadStatesIndex.value, 0x0)
        if cur_map_id == 0xCB:  # hideout elevator
            # need to get lift key to auto lift to b4f
            # if no silph scope, set exit warp to b4f
            if 0x4A in self.get_items_in_bag():
                if 0x48 not in self.get_items_in_bag():
                    for i in range(2):
                        self.pyboy.set_memory_value(RAM.wWarpEntries.value + (i * 4) + 2, 2)  # warp id
                        self.pyboy.set_memory_value(RAM.wWarpEntries.value + (i * 4) + 3, 0xCA)  # warp map id
                else:
                    # else set exit warp to b1f
                    for i in range(2):
                        self.pyboy.set_memory_value(RAM.wWarpEntries.value + (i * 4) + 2, 4)  # warp id
                        self.pyboy.set_memory_value(RAM.wWarpEntries.value + (i * 4) + 3, 0xC7)  # warp map id
        if cur_map_id == 0xEC:  # silph co elevator
            # no key_card, lift exit warp = 5f
            # have key_card, lift exit warp = 3f
            # have key_card, cleared masterball event, = 1f?
            if 0x30 not in self.get_items_in_bag():
                for i in range(2):  # warp to 5f
                    self.pyboy.set_memory_value(RAM.wWarpEntries.value + (i * 4) + 2, 2)  # warp id
                    self.pyboy.set_memory_value(RAM.wWarpEntries.value + (i * 4) + 3, 0xD2)  # warp map id
            elif not self.read_bit(0xD838, 5):  # not cleared masterball event
                for i in range(2):  # warp to 3f
                    self.pyboy.set_memory_value(RAM.wWarpEntries.value + (i * 4) + 2, 2)  # warp id
                    self.pyboy.set_memory_value(RAM.wWarpEntries.value + (i * 4) + 3, 0xD0)  # warp map id
            else:
                for i in range(2):  # warp to 1f
                    self.pyboy.set_memory_value(RAM.wWarpEntries.value + (i * 4) + 2, 3)  # warp id
                    self.pyboy.set_memory_value(RAM.wWarpEntries.value + (i * 4) + 3, 0xB5)  # warp map id
        if self.stage_manager.stage == 10:
            map_id = self.current_map_id - 1
            has_secret_key = 0x2B in self.get_items_in_bag()
            if not has_secret_key:
                # no key
                if map_id == 0xD8:
                    # in b1f
                    switch_on = self.read_bit(0xD796, 0)
                    if 'last_switch_on' not in self.secret_switch_states:
                        self.secret_switch_states['last_switch_on'] = switch_on
                    switches = [
                        (18, 26),
                        (20, 4)
                    ]
                    if 'given_reward' not in self.secret_switch_states:
                        self.secret_switch_states['given_reward'] = 0
                    last_given_reward = self.secret_switch_states['given_reward']
                    x, y = self.current_coords
                    if (x, y) == switches[1]:
                        # in b1f and no key and in front of switch 2
                        self.secret_switch_states['given_reward'] = 5 if last_given_reward != 8 else 8
                        # switch_on and 
                        facing_up = self.read_m(0xC109) == 4
                        if facing_up:
                            self.secret_switch_states['given_reward'] = 6 if last_given_reward != 8 else 8
                            if not self.is_in_battle() and self.read_m(0xFFB0) == 0:
                                # in menu and not in battle
                                self.secret_switch_states['given_reward'] = 7 if last_given_reward != 8 else 8
                            if self.secret_switch_states['last_switch_on'] != switch_on:
                                if not switch_on:
                                    # switch off
                                    self.secret_switch_states['given_reward'] = 7
                                else:
                                    # switch on
                                    self.secret_switch_states['given_reward'] = 8
                    elif last_given_reward == 8:
                        pass
                    elif last_given_reward != 8 and last_given_reward > 4:
                        self.secret_switch_states['given_reward'] = 4
                    elif (x, y) == switches[0]:
                        facing_up = self.read_m(0xC109) == 4
                        self.secret_switch_states['given_reward'] = 1 if last_given_reward != 4 else 4
                        if facing_up:
                            self.secret_switch_states['given_reward'] = 2 if last_given_reward != 4 else 4
                            if not self.is_in_battle() and self.read_m(0xFFB0) == 0:
                                # in menu and not in battle
                                self.secret_switch_states['given_reward'] = 3 if last_given_reward != 4 else 4
                            if self.secret_switch_states['last_switch_on'] != switch_on:
                                if not switch_on:
                                    # switch off
                                    self.secret_switch_states['given_reward'] = 4
                                else:
                                    # switch on
                                    self.secret_switch_states['given_reward'] = 3
                    elif self.secret_switch_states['given_reward'] != 4:
                        self.secret_switch_states['given_reward'] = 0
                    self.secret_switch_states['last_switch_on'] = switch_on
                    # if self.step_count % 5 == 0:
                    #     print(f'switch states: {self.secret_switch_states}')
                else:
                    # not in map and no key
                    self.secret_switch_states = {}
                    # if self.step_count % 5 == 0:
                    #     print(f'switch states: {self.secret_switch_states}')

    def replace_item_in_bag(self, source_item, target_item):
        items = self.get_items_in_bag()
        if source_item not in items:
            print(f'Warning: source item {source_item} not in bag')
            return
        if target_item in items:
            print(f'Warning: target item {target_item} already in bag')
            return
        all_items_quantity = self.get_items_quantity_in_bag()
        idx = items.index(source_item)
        cur_item_quantity = all_items_quantity[idx]
        self.pyboy.set_memory_value(0xD31E + idx*2, target_item)
        self.pyboy.set_memory_value(0xD31F + idx*2, cur_item_quantity)
        self._items_in_bag = None

    @staticmethod
    def set_bit(value, bit):
        return value | (1<<bit)
    
    def update_stage_manager(self):
        current_states = {
            'items': self.get_items_in_bag(),
            'map_id': self.current_map_id - 1,
            'badges': self.get_badges(),
            'visited_pokecenters': self.visited_pokecenter_list,
            'last_pokecenter': self.get_last_pokecenter_id(),
        }
        if 'events' in STAGE_DICT[self.stage_manager.stage]:
            event_list = STAGE_DICT[self.stage_manager.stage]['events']
            if 'EVENT_GOT_MASTER_BALL' in event_list:
                # EVENT_GOT_MASTER_BALL
                current_states['events'] = {'EVENT_GOT_MASTER_BALL': self.read_bit(0xD838, 5)}
            if 'CAN_USE_SURF' in event_list:
                # CAN_USE_SURF
                current_states['events'] = {'CAN_USE_SURF': self.can_use_surf}
        # if self.stage_manager.stage == 7:
        #     # EVENT_GOT_MASTER_BALL
        #     current_states['events'] = {'EVENT_GOT_MASTER_BALL': self.read_bit(0xD838, 5)}
        # elif self.stage_manager.stage == 9:
        #     # CAN_USE_SURF
        #     current_states['events'] = {'CAN_USE_SURF': self.can_use_surf}
        self.stage_manager.update(current_states)
        
        # additional blockings for stage 10
        if self.stage_manager.stage == 10:
            map_id = self.current_map_id - 1
            if map_id == 0xD8:
                # pokemon mansion b1f
                # if map_id not in self.hideout_elevator_maps:
                #     self.hideout_elevator_maps.append(map_id)
                bag_items = self.get_items_in_bag()
                additional_blocking = ['POKEMON_MANSION_B1F', 'POKEMON_MANSION_1F@6']
                if 0x2B not in bag_items:
                    # secret key not in bag items
                    # add blocking
                    if additional_blocking not in self.stage_manager.blockings:
                        self.stage_manager.blockings.append(additional_blocking)
                else:
                    # secret key in bag items
                    # remove blocking
                    if self.read_bit(0xD796, 0) is True:
                        # if switch on then remove blocking to exit
                        if additional_blocking in self.stage_manager.blockings:
                            self.stage_manager.blockings.remove(additional_blocking)
                    else:
                        # if switch off then add blocking to exit
                        if additional_blocking not in self.stage_manager.blockings:
                            self.stage_manager.blockings.append(additional_blocking)
            # # if have key card, can go to 3f
            # # if have master ball, can go to 1f
            # if 0x30 in self.get_items_in_bag():
            #     for i in range(2):  # warp to 3f
            #         self.pyboy.set_memory_value(RAM.wWarpEntries.value + (i * 4) + 2, 2)
        return current_states  # for debugging
    
    def _calculate_individual_stats(self, base_stats, level, ev, iv, is_hp=False):
        # hp = floor(((base_stats + iv) * 2 + sqrt(ev) / 4) * level / 100) + level + 10
        # other stats = floor(((base_stats + iv) * 2 + sqrt(ev) / 4) * level / 100) + 5
        if is_hp:
            return int(((base_stats + iv) * 2 + math.sqrt(ev) / 4) * level / 100) + level + 10
        else:
            return int(((base_stats + iv) * 2 + math.sqrt(ev) / 4) * level / 100) + 5
    
    def calculate_pokemon_stats(self, stats_dict):
        if 'species' not in stats_dict or stats_dict['species'] not in ID_TO_SPECIES or ID_TO_SPECIES[stats_dict['species']] not in BASE_STATS:
            print(f'\nERROR: error while calculating pokemon stats, invalid species: {stats_dict["species"]}')
            return {'max_hp': 10, 'atk': 5, 'def': 5, 'spd': 15, 'spc': 20}
        base_stats = BASE_STATS[ID_TO_SPECIES[stats_dict['species']]]
        # base_stats contains hp, atk, def, spd, spc
        stats_dict['atk_iv'] = stats_dict['atk_def_iv'] >> 4
        stats_dict['def_iv'] = stats_dict['atk_def_iv'] & 0xF
        stats_dict['spd_iv'] = stats_dict['spd_spc_iv'] >> 4
        stats_dict['spc_iv'] = stats_dict['spd_spc_iv'] & 0xF
        def get_hp_iv(stat_dict):
            hp_iv = 0
            hp_iv += 8 if stat_dict['atk_iv'] % 2 == 1 else 0
            hp_iv += 4 if stat_dict['def_iv'] % 2 == 1 else 0
            hp_iv += 2 if stat_dict['spd_iv'] % 2 == 1 else 0
            hp_iv += 1 if stat_dict['spc_iv'] % 2 == 1 else 0
            return hp_iv
        stats_dict['hp_iv'] = get_hp_iv(stats_dict)

        # calculate max_hp, atk, def, spd, spc
        stats_dict['max_hp'] = self._calculate_individual_stats(base_stats['hp'], stats_dict['level'], stats_dict['hp_ev'], stats_dict['hp_iv'], is_hp=True)
        stats_dict['atk'] = self._calculate_individual_stats(base_stats['atk'], stats_dict['level'], stats_dict['atk_ev'], stats_dict['atk_iv'])
        stats_dict['def'] = self._calculate_individual_stats(base_stats['def'], stats_dict['level'], stats_dict['def_ev'], stats_dict['def_iv'])
        stats_dict['spd'] = self._calculate_individual_stats(base_stats['spd'], stats_dict['level'], stats_dict['spd_ev'], stats_dict['spd_iv'])
        stats_dict['spc'] = self._calculate_individual_stats(base_stats['spc'], stats_dict['level'], stats_dict['spc_ev'], stats_dict['spc_iv'])
        return stats_dict

    def scripted_party_management(self):
        # party management
        # compare party lowest level with box highest level
        # if party lowest level < box highest level, swap
        party_addr_start = 0xD16B
        party_nicknames_addr_start = 0xd2b5
        party_species_addr_start = 0xD164
        box_mon_addr_start = 0xda96
        box_species_addr_start = 0xDA81
        box_nicknames_addr_start = 0xde06
        num_mon_in_box = self.num_mon_in_box
        # # level addr: party addr + 33 = box addr + 3

        if num_mon_in_box == 0:
            # no pokemon in box, do nothing
            return
        
        party_count = self.read_num_poke()
        if party_count < 6:
            # party not full, do nothing
            return
        
        party_levels = [self.read_m(party_addr_start + i * 44 + 33) for i in range(party_count)]
        lowest_party_level = min(party_levels)
        box_levels = [self.read_m(box_mon_addr_start + i * 33 + 3) for i in range(num_mon_in_box)]
        highest_box_level = max(box_levels)

        if highest_box_level <= lowest_party_level:
            # box pokemon not higher level than party pokemon
            # delete box pokemons to make space
            self.delete_box_pokemon_with_low_level(lowest_party_level)
            return
        
        # swap party pokemon with box pokemon
        # find the lowest level party pokemon
        lowest_party_level_idx = party_levels.index(lowest_party_level)
        # find the highest level box pokemon
        highest_box_level_idx = box_levels.index(highest_box_level)

        # # party mon address is slightly different, a few duplicates
        # # party mon addr + 33 = box mon addr + 3
        # # party mon addr + 34 / 35 = box mon addr + 1 / 2

        # last 5 stats are hp, atk, def, spd, spc, they are calculated from iv, ev, level
        # box pokemon does not have these stats, so need to calculate them
        named_idx = [[0, 'species', 'single'], [1, 'current_hp', 'double'], [3, 'level', 'single'], [14, 'exp', 'triple'], [17, 'hp_ev', 'double'], [19, 'atk_ev', 'double'], [21, 'def_ev', 'double'], [23, 'spd_ev', 'double'], [25, 'spc_ev', 'double'], [27, 'atk_def_iv', 'single'], [28, 'spd_spc_iv', 'single']]
        # [33, 'actual_level', 'single'] , [34, 'max_hp', 'double'], [36, 'atk', 'double'], [38, 'def', 'double'], [40, 'spd', 'double'], [42, 'spc', 'double']

        # # party pokemon
        # party_stats_dict = {}
        # index = lowest_party_level_idx
        # for idx, name, m_type in named_idx:
        #     if m_type == 'double':
        #         print(f'{name}: {self.read_double(party_addr_start + index * 44 + idx)}')
        #         party_stats_dict[name] = self.read_double(party_addr_start + index * 44 + idx)
        #     elif m_type == 'triple':
        #         print(f'{name}: {self.read_triple(party_addr_start + index * 44 + idx)}')
        #         party_stats_dict[name] = self.read_triple(party_addr_start + index * 44 + idx)
        #     else:
        #         print(f'{name}: {self.read_m(party_addr_start + index * 44 + idx)}')
        #         party_stats_dict[name] = self.read_m(party_addr_start + index * 44 + idx)

        # box pokemon
        box_stats_dict = {}
        index = highest_box_level_idx
        for idx, name, m_type in named_idx:
            if m_type == 'double':
                # print(f'{name}: {self.read_double(box_mon_addr_start + index * 33 + idx)}')
                box_stats_dict[name] = self.read_double(box_mon_addr_start + index * 33 + idx)
            elif m_type == 'triple':
                # print(f'{name}: {self.read_triple(box_mon_addr_start + index * 33 + idx)}')
                box_stats_dict[name] = self.read_triple(box_mon_addr_start + index * 33 + idx)
            else:
                # print(f'{name}: {self.read_m(box_mon_addr_start + index * 33 + idx)}')
                box_stats_dict[name] = self.read_m(box_mon_addr_start + index * 33 + idx)

        lowest_party_species = self.read_m(party_species_addr_start + lowest_party_level_idx)
        highest_box_species = self.read_m(box_species_addr_start + highest_box_level_idx)
        if lowest_party_species in ID_TO_SPECIES and highest_box_species in ID_TO_SPECIES:
            print(f'\nSwapping pokemon {ID_TO_SPECIES[lowest_party_species]} lv {lowest_party_level} with {ID_TO_SPECIES[highest_box_species]} lv {highest_box_level}')
        else:
            print(f'\nSwapping pokemon {lowest_party_species} lv {lowest_party_level} with {highest_box_species} lv {highest_box_level}')
        self.use_pc_swap_count += 1
        # calculate box pokemon stats
        box_stats_dict = self.calculate_pokemon_stats(box_stats_dict)

        # swap party pokemon with box pokemon
        # copy species
        self.pyboy.set_memory_value(party_species_addr_start + lowest_party_level_idx, self.read_m(box_species_addr_start + highest_box_level_idx))

        # copy all 0 to 33 from box to party
        for i in range(33):
            self.pyboy.set_memory_value(party_addr_start + lowest_party_level_idx * 44 + i, self.read_m(box_mon_addr_start + highest_box_level_idx * 33 + i))
            if i == 3:
                # copy level from box to party
                self.pyboy.set_memory_value(party_addr_start + lowest_party_level_idx * 44 + 33, self.read_m(box_mon_addr_start + highest_box_level_idx * 33 + 3))

        # copy the remaining stats from box to party
        # max_hp, atk, def, spd, spc
        box_stats = [box_stats_dict['max_hp'], box_stats_dict['atk'], box_stats_dict['def'], box_stats_dict['spd'], box_stats_dict['spc']]
        for i in range(5):
            # these stats are splitted into 2 bytes
            # first byte is the higher byte, second byte is the lower byte
            self.pyboy.set_memory_value(party_addr_start + lowest_party_level_idx * 44 + 34 + i * 2, box_stats[i] >> 8)
            self.pyboy.set_memory_value(party_addr_start + lowest_party_level_idx * 44 + 35 + i * 2, box_stats[i] & 0xFF)

        # copy nickname
        for i in range(11):
            self.pyboy.set_memory_value(party_nicknames_addr_start + lowest_party_level_idx * 11 + i, self.read_m(box_nicknames_addr_start + highest_box_level_idx * 11 + i))

        self.delete_box_pokemon(highest_box_level_idx, num_mon_in_box)
        self._num_mon_in_box = None

        self.delete_box_pokemon_with_low_level(lowest_party_level)


    def delete_box_pokemon_with_low_level(self, lowest_party_level):
        box_mon_addr_start = 0xda96
        # delete all box pokemon with level < party lowest level
        # step 1 find all box pokemon with level < party lowest level
        # step 2 delete them
        # step 3 update num_mon_in_box
        num_mon_in_box = self.num_mon_in_box
        if num_mon_in_box == 0:
            # no pokemon in box, do nothing
            return
        box_levels = [self.read_m(box_mon_addr_start + i * 33 + 3) for i in range(num_mon_in_box)]
        box_levels_to_delete = [i for i, x in enumerate(box_levels) if x <= lowest_party_level]
        # start from the last index to delete
        for i in range(len(box_levels_to_delete) - 1, -1, -1):
            self.delete_box_pokemon(box_levels_to_delete[i], num_mon_in_box)
            self._num_mon_in_box = None
            num_mon_in_box = self.num_mon_in_box

    def delete_box_pokemon(self, box_mon_idx, num_mon_in_box):
        box_mon_addr_start = 0xda96
        box_species_addr_start = 0xDA81
        box_nicknames_addr_start = 0xde06
        # delete the box pokemon by shifting the rest up
        # box mon only has 33 stats
        for i in range(box_mon_idx, num_mon_in_box - 1):
            # species
            self.pyboy.set_memory_value(box_species_addr_start + i, self.read_m(box_species_addr_start + i + 1))
            # stats
            for j in range(33):
                self.pyboy.set_memory_value(box_mon_addr_start + i * 33 + j, self.read_m(box_mon_addr_start + (i + 1) * 33 + j))
            # nickname
            for j in range(11):
                self.pyboy.set_memory_value(box_nicknames_addr_start + i * 11 + j, self.read_m(box_nicknames_addr_start + (i + 1) * 11 + j))
                
        # reduce num_mon_in_box by 1
        self.pyboy.set_memory_value(0xda80, num_mon_in_box - 1)
        # set the last box pokemon species to ff as it is empty
        self.pyboy.set_memory_value(box_species_addr_start + (num_mon_in_box - 1), 0xff)
        # set the last box pokemon stats to 0
        for i in range(33):
            self.pyboy.set_memory_value(box_mon_addr_start + (num_mon_in_box - 1) * 33 + i, 0)
        # set the last box pokemon nickname to 0
        for i in range(11):
            self.pyboy.set_memory_value(box_nicknames_addr_start + (num_mon_in_box - 1) * 11 + i, 0)

    def scripted_roll_party(self):
        # swap party pokemon order
        # by rolling 1 to last, 2 to 1, 3 to 2, 4 to 3, 5 to 4, 6 to 5 according to party pokemon count
        party_count = self.read_num_poke()
        if party_count < 2:
            # party not full, do nothing
            return
        
        if self.is_in_battle():
            # do not roll party during battle
            return
        
        if self.read_m(0xFFB0) == 0:  # hWY in menu
            # do not roll party during menu
            return
        party_addr_start = 0xD16B
        party_nicknames_addr_start = 0xd2b5
        party_species_addr_start = 0xD164  # 6 bytes for 6 party pokemon

        # copy the first pokemon to tmp
        tmp_species = self.read_m(party_species_addr_start)
        tmp_stats = []
        for i in range(44):
            tmp_stats.append(self.read_m(party_addr_start + i))
        tmp_nickname = []
        for i in range(11):
            tmp_nickname.append(self.read_m(party_nicknames_addr_start + i))

        # copy the rest of the pokemon to the previous one
        for i in range(party_count - 1):
            # species
            self.pyboy.set_memory_value(party_species_addr_start + i, self.read_m(party_species_addr_start + i + 1))
            # stats
            for j in range(44):
                self.pyboy.set_memory_value(party_addr_start + i * 44 + j, self.read_m(party_addr_start + (i + 1) * 44 + j))
            # nickname
            for j in range(11):
                self.pyboy.set_memory_value(party_nicknames_addr_start + i * 11 + j, self.read_m(party_nicknames_addr_start + (i + 1) * 11 + j))

        # copy the tmp to the last pokemon
        # species
        self.pyboy.set_memory_value(party_species_addr_start + party_count - 1, tmp_species)
        # stats
        for i in range(44):
            self.pyboy.set_memory_value(party_addr_start + (party_count - 1) * 44 + i, tmp_stats[i])
        # nickname
        for i in range(11):
            self.pyboy.set_memory_value(party_nicknames_addr_start + (party_count - 1) * 11 + i, tmp_nickname[i])

    def scripted_level_manager(self):
        # current_level = 0
        selected_level = LEVELS[self.current_level]
        # if self.current_level == 2:
        #     if self.stage_manager.stage > 7:
        #         if self.get_badges() >= 5:
        #             self.level_completed_skip_type = 1
        #             return True
        if 'badge' in selected_level:
            if self.get_badges() < selected_level['badge']:
                # not enough badge
                return False
        if 'event' in selected_level:
            if selected_level['event'] == 'CHAMPION':
                # D867, bit 1
                if not self.read_bit(0xD867, 1):
                    # not beaten champion yet
                    return False
        if 'last_pokecenter' in selected_level:
            found = False
            for pokecenter in selected_level['last_pokecenter']:
                if POKECENTER_TO_INDEX_DICT[pokecenter] == self.get_last_pokecenter_id():
                    found = True
                    break
            if not found:
                # not in the last pokecenter
                return False
        # if reached here, means all conditions met
        return True
        # if not self.level_manager_eval_mode:
        #     # if training mode, then return True
        #     return True
        # else:
        #     # if eval mode, increase current_level
        #     self.current_level += 1
        #     return True

