# BOEY's Baselines v2
This baseline beats the game in 800m steps with more than 80% success rate, further training to 1.2m steps can beat the game consistently with more than 97% success rate.

## Showcase
Videos of the agent beating the game can be found in the discord channel, it will be uploaded here soon.

## Neural Network Architecture
<img src="../../assets/Pokemon-Network-Architecture-Boey.png?raw=true">

(Right click and open image in a new tab to see the full size)
1. The neural network architecture is largely the same as my previous published network diagram, no LSTM, just FC (1024x1024)
    - 1a. Added new inputs of Nx9x10 for walkability, 3x ledges, water, warps, sprites, seen tiles*
    - Please refer to [previous version `boey_baselines`](../boey_baselines/README.md) for more details on the network architecture, observation space and rewards.

<sub>*Seen tiles show the 9x10 grid of tiles walked recently</sub>


## Game Mechanic Simplification
Because of partial observability of the environment (if we only use pixels), and without LSTM, the agent will need a lot of additional information when navigating the environment, especially menu navigation.

Instead of adding tons of additional information to the observation, I decided to simplify part of the environment to make it easier for the agent to learn as a baseline.

1. [S1] No Start button, so no Start menu, thus the neccessary function of the Start menu are scripted below.
2. [S2] Using cut, surf, pokeflute by press A on the correct tile, facing a tree tile / facing water or shore / facing snorlax sprite with pokeflute in bag, hm must be in bag and badge requirement must be fulfilled, no menu selection is needed, no learning hm move
3. [S3] No direct PC access, both Bill and Red PC, so cannot deposit, withdraw or release pokemon and items.
    - Pressing A infront of a PC will auto swap the highest level of Pokemon in box with the lowest level of Pokemon in party.
4. [S4] No direct Pokemart access, so cannot buy or sell items.
    - Pressing A infront of pokemart clerk will auto sell all the sellable items except potions and pokeballs, and buy up to 10x best Potions, Pokeballs offered by pokemart, if agent has enough money.
5. [S5] Additional SWAP_MON action is given to the agent, move pokemon position in party in a "clockwise" fashion, this is to allow the agent to switch pokemon the first pokemon for battle
6. [S6] Learning of new moves through leveling is scripted based on Power / PP / Additional effects.


## Puzzle Nagivation Simplification
This is the shorten the time needed to train an agent that can complete the game, I believe except N1, all the other simplifications in this category can be removed and still achieve the same result with at least 5x amount of training resources given.
1. [N1] No Strength hm is needed, all strength puzzle event flags are set to completed, this affects Seaform Island and Victory Road.
2. [N2] Safari zone has unlimited steps, and it is free to enter when the agent ran out of money to prevent hardstuck.
3. [N3] No lift menu access for Celadon Hideout and Saffron Silph Co., it is scripted to exit to the correct floor after the agent obtained Lift Key / Secret Card.
4. [N4] Spinner tiles nerfed to push back only 1 tile.
5. [N5] Restricted access to certain area to shrink the area needed to be observed to progress the game, managed by Stage Manager, such as:
    1. Block access to Diglett's Cave, to prevent unnecessary backtracking.
    2. Unable to enter Pokemon Tower before getting Silph Scope
    3. Blocked access to next route if Gym badge is obtainable in this city/town but is not obtained, to prevent skipping of gyms.
6. [N6] Additional reward given to solve Pokemon Mansion puzzle secret switches.


## Optimization
1. [O1] Text speed is always instant, no battle animation
2. [O2] Instead of a fixed 24 frames per action, the env will auto step with button A if in battle and no action input is needed.
3. [O3] Stage Manager, (it is called `level_manager` in the code), instead of always starting from the initial state, half of the envs will always start from the same `Stage` and the other envs will reset to a recent checkpoint of the worse performing `Stage` of the game. This allow the agent to learn the harder part of the game faster.


## Unintended Increase of Difficulty
1. [D1] Cannot use items out of battle, such as potions, revives, pp restores, repels, escape rope, rare candies, evolution stones etc.
2. [D2] No HM and TM moves, only have access to moves learned through leveling up, movesets through leveling are incredibly bad in Gen 1.


## Training parameters
1. [T1] Total timesteps: 800 millions
2. [T2] Learning Rate: 3e-4
3. [T3] Epoch: 3 (< 100m Total Timestep) -> 1
4. [T4] Batch size: 2048
5. [T5] Gamma: 0.999 (< 400m) -> 0.9996
6. [T6] Entropy Coefficient: 0.01
7. [T7] Early termination of env episode: No reward obtained for the last 10k steps.
8. [T8] N envs: 48 envs
