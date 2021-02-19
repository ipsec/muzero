from pathlib import Path

import tensorflow as tf

data_path = Path("./data/muzero")
data_path.mkdir(parents=True, exist_ok=True)
summary_writer = tf.summary.create_file_writer(str(data_path) + "/summary/")


def write_summary(episode, loss):
    with summary_writer.as_default():
        tf.summary.scalar("Loss", loss, step=episode)
