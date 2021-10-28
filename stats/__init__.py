from pathlib import Path

import ray
import tensorflow as tf


@ray.remote
class Summary(object):
    def __init__(self):
        self.data_path = Path("./summary")
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.summary_writer = tf.summary.create_file_writer(str(self.data_path))
        self._loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        self._reward = tf.keras.metrics.Mean('reward', dtype=tf.float32)
        self._games = tf.keras.metrics.Sum('games', dtype=tf.int32)

    def publish_loss(self, step: int = 0):
        with self.summary_writer.as_default():
            tf.summary.scalar("Loss", self._loss.result(), step)
        self._loss.reset_states()

    def publish_games(self, step):
        with self.summary_writer.as_default():
            tf.summary.scalar("Games", self._games.result(), step)
            tf.summary.scalar("Reward Mean", self._reward.result(), step)
        # self._games.reset_states()
        self._reward.reset_states()

    def games(self):
        self._games(1)

    def reward(self, value: float):
        self._reward(value)

    def loss(self, value: float):
        self._loss(value)
