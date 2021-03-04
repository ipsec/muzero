import logging
from pathlib import Path
from threading import Thread

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from tensorflow.keras.optimizers import Adam

from games.game import make_atari_config, ReplayBuffer
from muzero import play_game, train_network, save_checkpoints
from storage import SharedStorage
# get root logger
from summary import write_summary

log = logging.getLogger(__name__)

app = FastAPI()

STATIC_PATH = Path("static")
TEMPLATES_PATH = Path("templates")
Path.mkdir(STATIC_PATH, parents=True, exist_ok=True)
Path.mkdir(TEMPLATES_PATH, parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_PATH), name="static")
templates = Jinja2Templates(directory=TEMPLATES_PATH)

muzero_config = make_atari_config()
storage = SharedStorage(muzero_config)
replay_buffer = ReplayBuffer(muzero_config)
optimizer = Adam(learning_rate=0.01)

@app.get('/')
def root():
    return {"success": True}


@app.post('/train')
async def train():
    count = 0
    while True:
        for _ in range(muzero_config.num_games):
            game = play_game(muzero_config, storage.latest_network())
            write_summary(count, np.sum(game.rewards))
            replay_buffer.save_game(game)
            count += 1

        if len(replay_buffer.buffer) >= muzero_config.batch_size:
            train_network(muzero_config, storage, replay_buffer, optimizer)
            save_checkpoints(storage.latest_network())

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
