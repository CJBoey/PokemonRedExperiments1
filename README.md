# Train RL agents to play Pokemon Red

## First RL agent to beat Pokemon Red! 🎉
### Watch the Video on Youtube! 
<a href="http://www.youtube.com/watch?feature=player_embedded&v=MEQWO9CSJDo
" target="_blank"><img src="http://img.youtube.com/vi/MEQWO9CSJDo/0.jpg" 
alt="From Pixels to Pokedex: How This RL Agent Conquered Pokemon Red! Ep1" height="180" /></a>
<a href="http://www.youtube.com/watch?feature=player_embedded&v=_CqO5yhYv1E
" target="_blank"><img src="http://img.youtube.com/vi/_CqO5yhYv1E/0.jpg" 
alt="From Pixels to Pokedex: How This RL Agent Conquered Pokemon Red! Ep2" height="180" /></a>
<p><sub>*Click to play the video</sub></p>
<p>The code to reproduce the results in the video can be found in `<a href="/baselines/boey_baselines2/">baselines/boey_baselines2</a>` directory.</p>

##
<p float="left">
  <a href="https://youtu.be/DcYLT37ImBY">
    <img src="/assets/youtube.jpg?raw=true" height="192">
  </a>
  <a href="https://youtu.be/DcYLT37ImBY">
    <img src="/assets/poke_map.gif?raw=true" height="192">
  </a>
</p>

## Join the discord server
[![Join the Discord server!](https://invidget.switchblade.xyz/RvadteZk4G)](http://discord.gg/RvadteZk4G)
  
## Running the Pretrained Model Interactively 🎮  
🐍 Python 3.10 is recommended. Other versions may work but have not been tested.   
You also need to install ffmpeg and have it available in the command line.

1. Copy your legally obtained Pokemon Red ROM into the base directory. You can find this using google, it should be 1MB. Rename it to `PokemonRed.gb` if it is not already. The sha1 sum should be `ea9bcae617fdf159b045185467ae58b2e4a48b9a`, which you can verify by running `shasum PokemonRed.gb`. 
2. Move into the `baselines/` directory:  
 ```cd baselines```  
3. Install dependencies:  
```pip install -r requirements.txt```  
It may be necessary in some cases to separately install the SDL libraries.  
4. Run:  
```python run_pretrained_interactive.py```
  
Interact with the emulator using the arrow keys and the `a` and `s` keys (A and B buttons).  
You can pause the AI's input during the game by editing `agent_enabled.txt`

Note: the Pokemon.gb file MUST be in the main directory and your current directory MUST be the `baselines/` directory in order for this to work.

## Training the Model 🏋️ 

<img src="/assets/grid.png?raw=true" height="156">

### 10-21-23: Updated Version! 

This version still needs some tuning, but it can clear the first gym in a small fraction of the time and compute resources. It can work with as few as 16 cores and ~20G of RAM. This is the place for active development and updates! 

1. Previous steps 1-3
2. Run:  
```python run_baseline_parallel_fast.py```

## Tracking Training Progress 📈 
The current state of each game is rendered to images in the session directory.   
You can track the progress in tensorboard by moving into the session directory and running:  
```tensorboard --logdir .```  
You can then navigate to `localhost:6006` in your browser to view metrics.  
To enable wandb integration, change `use_wandb_logging` in the training script to `True`.

## Extra 🐜
Map visualization code can be found in `visualization/` directory.

## Supporting Libraries
Check out these awesome projects!
### [PyBoy](https://github.com/Baekalfen/PyBoy)
<a href="https://github.com/Baekalfen/PyBoy">
  <img src="/assets/pyboy.svg" height="64">
</a>

### [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3)
<a href="https://github.com/DLR-RM/stable-baselines3">
  <img src="/assets/sblogo.png" height="64">
</a>
