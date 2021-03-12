from pathlib import Path

import tensorflow as tf

data_path = Path("./data/muzero")
data_path.mkdir(parents=True, exist_ok=True)
summary_writer = tf.summary.create_file_writer(str(data_path) + "/summary/")


def write_summary_score(score, step):
    with summary_writer.as_default():
        tf.summary.scalar("Score", score, step)
        tf.summary.flush()

