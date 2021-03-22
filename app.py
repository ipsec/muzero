import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path

import gym
import numpy as np
import uvicorn
from fastapi import FastAPI, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from tensorflow.keras.optimizers import Adam

from games.game import make_atari_config, ReplayBuffer
from muzero import update_weights, run_selfplay
from storage import SharedStorage

# get root logger

log = logging.getLogger(__name__)

app = FastAPI()

STATIC_PATH = Path("static")
TEMPLATES_PATH = Path("templates")
Path.mkdir(STATIC_PATH, parents=True, exist_ok=True)
Path.mkdir(TEMPLATES_PATH, parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_PATH), name="static")
templates = Jinja2Templates(directory=TEMPLATES_PATH)

env = gym.make('CartPole-v1')

muzero_config = make_atari_config(env=env)
storage = SharedStorage(muzero_config)
replay_buffer = ReplayBuffer(muzero_config)
optimizer = Adam(learning_rate=0.0001)


def play_game():
    run_selfplay(muzero_config, storage, replay_buffer)


def train(network):
    batch = replay_buffer.sample_batch(muzero_config.num_unroll_steps, muzero_config.td_steps)
    update_weights(optimizer, network, batch, muzero_config.weight_decay)


@app.get('/')
async def root():
    return {"success": True}


@app.post('/config/reload')
async def config_reload(game: str = 'CartPole-v1'):
    env = gym.make(game)
    muzero_config = make_atari_config(env=env)
    return {"success": True}


@app.post('/train')
async def train_network(background_tasks: BackgroundTasks):
    network = storage.latest_network()
    for i in range(muzero_config.training_steps):
        background_tasks.add_task(play_game)
        if i % muzero_config.checkpoint_interval == 0:
            storage.save_network(i, network)
        background_tasks.add_task(train, network)

    storage.save_network(muzero_config.training_steps, network)

    return {"success": True}


@app.get('/stats')
def stats():
    scores = 0.0

    if len(replay_buffer.buffer) > 0:
        scores = np.mean([np.sum(game.rewards) for game in replay_buffer.buffer])

    return {'num_games': muzero_config.window_size,
            'num_games_done': len(replay_buffer.buffer),
            'mean_scores': scores}


if __name__ == "__main__":
    uvicorn.run(app,
                host="0.0.0.0",
                port=8080,
                log_level=logging.INFO,
                access_log=True,
                use_colors=True)
