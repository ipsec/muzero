import collections
from typing import Optional, Union

import numpy as np
import tensorflow as tf

KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])


class MinMaxStats(object):
    """A class that holds the min-max values of the tree."""

    def __init__(self, known_bounds: Optional[KnownBounds]):
        self.maximum = known_bounds.max if known_bounds else -float('inf')
        self.minimum = known_bounds.min if known_bounds else float('inf')

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class Node(object):

    def __init__(self, prior: float):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


def tf_scalar_to_support(x: tf.Tensor, support_size: int):
    value = tf.math.sign(x) * (tf.math.sqrt(tf.math.abs(x) + 1) - 1) + tf.multiply(0.001, x)

    transformed = tf.clip_by_value(value, -support_size, support_size)

    floored = tf.floor(transformed)
    prob = transformed - floored

    idx_0 = tf.expand_dims(tf.cast(tf.squeeze(floored + support_size), dtype=tf.int32), -1)
    idx_1 = tf.expand_dims(tf.cast(tf.squeeze(floored + support_size + 1), dtype=tf.int32), -1)
    idx_0 = tf.stack([tf.range(x.shape[1]), idx_0])
    idx_1 = tf.stack([tf.range(x.shape[1]), idx_1])
    indexes = tf.squeeze(tf.stack([idx_0, idx_1]))

    updates = tf.squeeze(tf.concat([1 - prob, prob], axis=0))
    values = tf.scatter_nd(indexes, updates, (1, 2 * support_size + 1))
    return values


def tf_support_to_scalar(x: tf.Tensor, support_size: int):
    probabilities = tf.nn.softmax(x, axis=1)
    support = tf.constant(list(range(-support_size, support_size + 1)), dtype=tf.float32, shape=probabilities.shape)
    value = tf.reduce_sum(support * probabilities, axis=1, keepdims=True)
    value = tf.math.sign(value) * (
                ((tf.math.sqrt(1 + 4 * 0.001 * (tf.math.abs(value) + 1 + 0.001)) - 1) / (2 * 0.001)) ** 2 - 1)
    return value


def cast_to_tensor(x: Union[np.ndarray, float]) -> tf.Tensor:
    return tf.convert_to_tensor(x, dtype=tf.keras.backend.floatx())
