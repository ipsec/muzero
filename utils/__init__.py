import collections
from typing import Optional

import tensorflow as tf

MAXIMUM_FLOAT_VALUE = float('inf')

KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])


class MinMaxStats(object):
    """A class that holds the min-max values of the tree."""

    def __init__(self, known_bounds: Optional[KnownBounds]):
        self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
        self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE

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


def scalar_transform(x: tf.Tensor, eps: float = 0.001) -> tf.Tensor:
    return tf.math.sign(x) * (tf.math.sqrt(tf.math.abs(x) + 1) - 1) + tf.multiply(eps, x)


def inverse_scalar_transform(x: tf.Tensor, eps: float = 0.001) -> tf.Tensor:
    return tf.math.sign(x) * (((tf.math.sqrt(1. + 4. * eps * (tf.math.abs(x) + 1 + eps)) - 1) / (2 * eps)) ** 2 - 1)


def tf_scalar_to_support(scalar: tf.Tensor, support_size: int) -> tf.Tensor:
    if support_size == 0:  # Simple regression (support in this case can be the mean of a Gaussian)
        return scalar

    scalar = scalar_transform(scalar)

    transformed = tf.clip_by_value(scalar, -support_size, support_size - 1e-6)
    floored = tf.floor(transformed)
    prob = transformed - floored

    idx_0 = tf.expand_dims(tf.cast(tf.squeeze(floored + support_size), dtype=tf.int32), -1)
    idx_1 = tf.expand_dims(tf.cast(tf.squeeze(floored + support_size + 1), dtype=tf.int32), -1)
    idx_0 = tf.stack([tf.range(scalar.shape[1]), idx_0])
    idx_1 = tf.stack([tf.range(scalar.shape[1]), idx_1])
    indexes = tf.squeeze(tf.stack([idx_0, idx_1]))

    updates = tf.squeeze(tf.concat([1 - prob, prob], axis=0))
    return tf.scatter_nd(indexes, updates, (1, 2 * support_size + 1))


def tf_support_to_scalar(support: tf.Tensor, support_size: int) -> tf.Tensor:
    if support_size == 0:
        return support

    bins = tf.range(-support_size, support_size + 1, dtype=tf.float32)
    value = tf.tensordot(tf.squeeze(support), tf.squeeze(bins), 1)

    return inverse_scalar_transform(value)
