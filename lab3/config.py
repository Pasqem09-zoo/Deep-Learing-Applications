ENV_NAME = "CartPole-v1"

SEED = 3097

EPISODES = 500
MAX_STEPS = 1000

GAMMA = 0.99
LEARNING_RATE = 0.001
HIDDEN_DIM = 128

CHECKPOINT_EVERY = 100

### ogni eval_every iterazioni faccio una valutazione dell'agente, runnando eval_episodes episodi nell'ambiente e calcolando la media dei reward ottenuti.
N = 50
M = 5

RENDER = True
RENDER_EPISODES = 5