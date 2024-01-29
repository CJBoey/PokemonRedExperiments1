from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Image
import numpy as np
from einops import rearrange
from .constants import LEVELS
import math
import os
from pathlib import Path

def merge_dicts_by_mean(dicts):
    sum_dict = {}
    count_dict = {}

    for d in dicts:
        for k, v in d.items():
            if isinstance(v, (int, float)): 
                sum_dict[k] = sum_dict.get(k, 0) + v
                count_dict[k] = count_dict.get(k, 0) + 1

    mean_dict = {}
    for k in sum_dict:
        mean_dict[k] = sum_dict[k] / count_dict[k]

    return mean_dict

class TensorboardCallback(BaseCallback):

    def __init__(self, verbose=0, save_state_dir=None):
        self.step_count = 0
        # self.ep_len1 = 0
        self.ep_len1 = 1
        # self.gap = 40960 // 32
        # self.ep_len1 = 40960 * 2
        self.gap = 10240
        self.save_state_dir = save_state_dir
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # check the first env only
        # if self.training_env.env_method("check_if_done", indices=[0])[0]:
        if self.step_count and self.step_count % self.gap == 0:
            all_infos = self.training_env.get_attr("current_stats_with_id")
            all_final_infos = []
            for stats in all_infos:
                if stats:
                    all_final_infos.append(stats)
            combined_stats = {}
            level_stats = {}
            level_dist = {}
            stats_key = ['step', 'rewards', 'eventr', 'levelr', 'op_lvlr', 'deadr', 'visited_pokecenterr', 'trees_cutr', 
                               'hmr', 'hm_usabler', 'special_key_itemsr', 'special_seen_coords_count', 'specialr', 'coord_count', 'perm_coord_count', 'seen_map_count', 'hp', 'pcount', 'stager', 'n_stage', 'healr',
                               'badger']
            for env_stats in all_final_infos:
                env_id = env_stats.pop("env_id", None)
                env_level = env_stats.get("current_level", None)
                for key,val in env_stats.items():
                    self.logger.record(f"env_stats/{env_id}/{key}", val)
                    if key in stats_key or \
                                'badge_' in key:
                        if key[:12] not in combined_stats:
                            combined_stats[key[:12]] = []
                        combined_stats[key[:12]].append(val)
                        if env_level is not None:
                            if env_level not in level_stats:
                                level_stats[env_level] = {}
                            if key[:12] not in level_stats[env_level]:
                                level_stats[env_level][key[:12]] = []
                            level_stats[env_level][key[:12]].append(val)
                if env_level is not None:
                    if str(env_level) not in level_dist:
                        level_dist[str(env_level)] = 0
                    level_dist[str(env_level)] += 1
            # plot combined stats, max min mean, and some useful stats
            for key, val in combined_stats.items():
                self.logger.record(f"env_stats/combined/{key}_mean", np.mean(val))
                self.logger.record(f"env_stats/combined/{key}_max", np.max(val))
                self.logger.record(f"env_stats/combined/{key}_min", np.min(val))

            # plot level stats, max min mean, and some useful stats
            for level, stats in level_stats.items():
                for key, val in stats.items():
                    self.logger.record(f"env_stats/level/{level}/{key}_mean", np.mean(val))
                    self.logger.record(f"env_stats/level/{level}/{key}_max", np.max(val))
                    self.logger.record(f"env_stats/level/{level}/{key}_min", np.min(val))

            # plot level count
            for level, count in level_dist.items():
                self.logger.record(f"env_stats/level/{level}/dist", count)

            # reduced the frequency of saving images for performance
            # if self.step_count % self.ep_len1 == 0:
            # use reduce_res=False for full res screens
            if self.step_count % (self.gap * 4) == 0:
                images = self.training_env.env_method("render", reduce_res=False)
                images_arr = np.array(images)
                images_row = rearrange(images_arr, "b h w c -> (b h) w c")
                self.logger.record("trajectory/image", Image(images_row, "HWC"), exclude=("stdout", "log", "json", "csv"))
        
        if self.save_state_dir and self.step_count and self.step_count % (self.gap // 4) == 0:
            # plot level success rate
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
                if level > len(LEVELS):
                    continue
                # if level > highest_level:
                num_states = len(list(level_dir.glob('*')))
                # level_states_ordered = sorted(list(level_dir.glob('*')), key=os.path.getmtime)
                # num_states = len(level_states_ordered)
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
                # self.level_in_charge = 0

                # placeholder for level_stats and level_selection_chance_list
                level_stats = {0: {'S': 0, 'F': 0, 'success_rate': 0}}
                level_stats_50 = {0: {'S': 0, 'F': 0, 'success_rate': 0}}
                level_selection_chance_list = [1]
                level_selection_chance_list_50 = [1]
                pass
            else:
                # check stats of level envs in save_state_dir / level_{level_in_charge}.txt
                # content of file is something like: SSSSFFS
                # S: success, F: failed
                # get the last 20 characters, count the number of S and F
                # assign at failure rate for each level env
                # the level env with highest failure rate will more likely to be assigned to assist env
                level_stats = {}
                level_stats_50 = {}
                for level in range(explored_levels):
                    level_stats[level] = {'S': 0, 'F': 0}
                    level_stats_50[level] = {'S': 0, 'F': 0}
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
                            for char in stats[-50:]:
                                level_stats_50[level][char] += 1
                
                # calculate failure rate
                for level in range(explored_levels):
                    if level_stats[level]['S'] + level_stats[level]['F'] == 0:
                        # insufficient stats, assign failure rate to 1 first
                        level_stats[level]['failure_rate'] = 1
                    else:
                        level_stats[level]['failure_rate'] = level_stats[level]['F'] / (level_stats[level]['S'] + level_stats[level]['F'])
                    # calculate failure rate for last 50 stats
                    if level_stats_50[level]['S'] + level_stats_50[level]['F'] == 0:
                        # insufficient stats, assign failure rate to 1 first
                        level_stats_50[level]['failure_rate'] = 1
                    else:
                        level_stats_50[level]['failure_rate'] = level_stats_50[level]['F'] / (level_stats_50[level]['S'] + level_stats_50[level]['F'])
                
                # convert failure_rate to success_rate for each level in level_stats
                # success_rate = 1 - failure_rate
                for level in range(explored_levels):
                    level_stats[level]['success_rate'] = 1 - level_stats[level]['failure_rate']
                    level_stats_50[level]['success_rate'] = 1 - level_stats_50[level]['failure_rate']
                        
                total_failure_rate = sum([level_stats[level]['failure_rate'] for level in range(explored_levels)])
                total_failure_rate_50 = sum([level_stats_50[level]['failure_rate'] for level in range(explored_levels)])
                level_selection_chance_list = [level_stats[level]['failure_rate'] / total_failure_rate for level in range(explored_levels)]
                level_selection_chance_list_50 = [level_stats_50[level]['failure_rate'] / total_failure_rate_50 for level in range(explored_levels)]
            # plot level success rate into env_stats/level/{level}/success_rate
            for level in range(explored_levels):
                self.logger.record(f"env_stats/level/{level}/success_rate", level_stats[level]['success_rate'])
                self.logger.record(f"env_stats/level/{level}/success_rate_50", level_stats_50[level]['success_rate'])
            # plot level selection chance into env_stats/level/{level}/selection_chance
            for level in range(explored_levels):
                self.logger.record(f"env_stats/level/{level}/selection_chance", level_selection_chance_list[level])
                self.logger.record(f"env_stats/level/{level}/selection_chance_50", level_selection_chance_list_50[level])
        self.step_count += 1


        return True


class GammaScheduleCallback(BaseCallback):

    def __init__(self, verbose=0, init_gamma=0.999, target_gamma=0.9999, given_timesteps=100_000_000, start_from=0):
        # self.step_count = 0
        self.init_gamma = init_gamma
        self.target_gamma = target_gamma
        self.given_timesteps = given_timesteps
        self.start_from = start_from
        super().__init__(verbose)

    def set_gamma(self, gamma):
        self.model.gamma = gamma
        self.model.rollout_buffer.gamma = gamma
    
    def _on_training_start(self) -> None:
        self.set_gamma(self.init_gamma)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        # self.num_timesteps  # multiple of self.model.n_steps
        # self.n_calls  # number of times _on_step is called
        init_n = 1 - self.init_gamma
        target_n = 1 - self.target_gamma
        target_mult = (init_n / target_n) - 1
        n = init_n / (1 + target_mult * min((self.num_timesteps + self.start_from) / self.given_timesteps, 1))
        gamma = 1 - n
        self.set_gamma(gamma)
        self.logger.record("train/gamma", self.model.gamma)
        return True
