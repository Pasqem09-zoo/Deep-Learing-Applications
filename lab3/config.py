# =============================
# General setup
# =============================
ENV_NAME = "CartPole-v1"      # "CartPole-v1" or "LunarLander-v3"
SEED = 3097
ALGORITHM = "a2c"             # "reinforce" or "a2c"


# =============================
# Common training parameters
# =============================
EPISODES = 3000               # number of training episodes
MAX_STEPS = 1000              # maximum steps per episode
GAMMA = 0.99                  # discount factor
HIDDEN_DIM = 128

CHECKPOINT_EVERY = 100        # save checkpoint every N episodes
N = 300                       # evaluate every N episodes (100)
M = 10                        # number of evaluation episodes


# =============================
# REINFORCE parameters
# =============================
LEARNING_RATE = 1e-3          # policy learning rate
VALUE_LEARNING_RATE = 1e-3    # value baseline learning rate
USE_VALUE_BASELINE = True     # use learned V(s) baseline


# =============================
# A2C parameters
# =============================
A2C_ACTOR_LR = 1e-3           # actor learning rate
A2C_CRITIC_LR = 1e-3          # critic learning rate
A2C_N_STEPS = 40              # rollout length before each update
A2C_ENTROPY_BETA = 0.005      # entropy bonus coefficient
A2C_GRAD_CLIP = 0.5           # gradient clipping threshold
A2C_ADV_NORM = True           # normalize advantages within rollout


# =============================
# Rendering
# =============================
RENDER = False                 # show final trained agent
RENDER_EPISODES = 8           # number of rendered episodes
EVAL_CARTPOLE_THETA = 0.418   # relaxed CartPole angle threshold


# =============================
# Weights & Biases
# =============================
WANDB_ENABLED = True          # enable W&B logging
WANDB_PROJECT = "dlaLab3"     # W&B project name
WANDB_RUN_NAME = "cartpole-a2c"