from pathlib import Path

import tensorflow as tf


def create_summary(name: str):
    data = Path(f"./summary/{name}")
    data.mkdir(parents=True, exist_ok=True)
    return tf.summary.create_file_writer(str(data))