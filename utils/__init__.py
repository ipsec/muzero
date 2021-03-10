import tensorflow as tf
import numpy as np
from dataclasses import dataclass
from typing import Optional

import typing

MAXIMUM_FLOAT_VALUE = float('inf')


@dataclass
class KnownBounds:
    min: float
    max: float


# KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])


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


"""
def cast_to_tensor(x: typing.Union[np.ndarray, float]) -> tf.Tensor:
    return tf.convert_to_tensor(x, dtype=tf.keras.backend.floatx())


def atari_reward_transform(x: tf.Tensor, eps: float = 0.001) -> tf.Tensor:
    return tf.math.sign(x) * (tf.math.sqrt(tf.math.abs(x) + 1) - 1) + tf.constant(eps) * x


def inverse_atari_reward_transform(x: tf.Tensor, eps: float = 0.001) -> tf.Tensor:
    return tf.math.sign(x) * (((tf.math.sqrt(1. + 4. * eps * (tf.math.abs(x) + 1 + eps)) - 1) / (2 * eps)) ** 2 - 1)


def support_to_scalar(x: tf.Tensor, support_size: int,
                      inv_reward_transformer: typing.Callable = inverse_atari_reward_transform, **kwargs) -> tf.Tensor:
    if support_size == 0:  # Simple regression (support in this case can be the mean of a Gaussian)
        return x

    bins = tf.range(-support_size, support_size + 1, dtype=tf.float32)
    y = tf.tensordot(tf.squeeze(x), tf.squeeze(bins), 1)

    value = inv_reward_transformer(y, **kwargs)

    return value


def scalar_to_support(x: tf.Tensor, support_size: int,
                      reward_transformer: typing.Callable = atari_reward_transform, **kwargs) -> tf.Tensor:
    if support_size == 0:  # Simple regression (support in this case can be the mean of a Gaussian)
        return x

    transformed = tf.clip_by_value(reward_transformer(x, **kwargs), -support_size, support_size - 1e-8)
    floored = tf.floor(transformed)  # Lower-bound support integer
    prob = transformed - floored  # Proportion between adjacent integers

    idx_0 = tf.expand_dims(tf.cast(tf.squeeze(floored + support_size), dtype=tf.int32), -1)
    idx_1 = tf.expand_dims(tf.cast(tf.squeeze(floored + support_size + 1), dtype=tf.int32), -1)
    idx_0 = tf.stack([tf.range(x.shape[1]), idx_0])
    idx_1 = tf.stack([tf.range(x.shape[1]), idx_1])
    idxs = tf.squeeze(tf.stack([idx_0, idx_1]))

    update_0 = tf.constant(1 - prob)
    update_1 = tf.constant(prob)

    updates = tf.squeeze(tf.concat([update_0, update_1], axis=0))
    res = None
    try:
        res = tf.scatter_nd(idxs, updates, (1, 2 * support_size + 1))
    except Exception as e:
        print(e)

    return res
"""


def atari_reward_transform(x: np.ndarray, var_eps: float = 0.001) -> np.ndarray:
    """
    Scalar transformation of rewards to stabilize variance and reduce scale.
    :references: https://arxiv.org/pdf/1805.11593.pdf
    """
    return np.sign(x) * (np.sqrt(np.abs(x) + 1) - 1) + var_eps * x


def inverse_atari_reward_transform(x: np.ndarray, var_eps: float = 0.001) -> np.ndarray:
    """
    Inverse scalar transformation of atari_reward_transform function as used in the canonical MuZero paper.
    :references: https://arxiv.org/pdf/1805.11593.pdf
    """
    return np.sign(x) * (((np.sqrt(1 + 4 * var_eps * (np.abs(x) + 1 + var_eps)) - 1) / (2 * var_eps)) ** 2 - 1)


def support_to_scalar(x: np.ndarray, support_size: int,
                      inv_reward_transformer: typing.Callable = inverse_atari_reward_transform, **kwargs) -> np.ndarray:
    """
    Recast distributional representation of floats back to floats. As the bins are symmetrically oriented around 0,
    this is simply done by taking a dot-product of the vector that represents the bins' integer range with the
    probability bins. After recasting of the floats, the floats are inverse-transformed for the scaling function.
    :param x: np.ndarray 2D-array of floats in distributional representation: len(scalars) x (support_size * 2 + 1)
    :param support_size:
    :param inv_reward_transformer: Inverse of the elementwise function that scales floats before casting them to bins.
    :param kwargs: Keyword arguments for inv_reward_transformer.
    :return: np.ndarray of size len(scalars) x 1
    """
    if support_size == 0:  # Simple regression (support in this case can be the mean of a Gaussian)
        return x

    bins = np.arange(-support_size, support_size + 1)
    y = np.dot(x, bins)

    value = inv_reward_transformer(y, **kwargs)

    return value


def scalar_to_support(x: np.ndarray, support_size: int,
                      reward_transformer: typing.Callable = atari_reward_transform, **kwargs) -> np.ndarray:
    """
    Cast a scalar or array of scalars to a distributional representation symmetric around 0.
    For example, the float 3.4 given a support size of 5 will create 11 bins for integers [-5, ..., 5].
    Each bin is assigned a probability value of 0, bins 4 and 3 will receive probabilities .4 and .6 respectively.
    :param x: np.ndarray 1D-array of floats to be cast to distributional bins.
    :param support_size: int Number of bins indicating integer range symmetric around zero.
    :param reward_transformer: Elementwise function to scale floats before casting them to bins.
    :param kwargs: Keyword arguments for reward_transformer.
    :return: np.ndarray of size len(x) x (support_size * 2 + 1)
    """
    if support_size == 0:  # Simple regression (support in this case can be the mean of a Gaussian)
        return x

    # Clip float to fit within the support_size. Values exceeding this will be assigned to the closest bin.
    transformed = np.clip(reward_transformer(x, **kwargs), a_min=-support_size, a_max=support_size - 1e-8)
    floored = np.floor(transformed).astype(int)  # Lower-bound support integer
    prob = transformed - floored  # Proportion between adjacent integers

    bins = np.zeros((len(x), 2 * support_size + 1))

    bins[np.arange(len(x)), floored + support_size] = 1 - prob
    bins[np.arange(len(x)), floored + support_size + 1] = prob

    return bins
