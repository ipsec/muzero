from pathlib import Path
from typing import List, Tuple, Callable

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.models import Model

from config import MuZeroConfig
from games.game import Action
from models import NetworkOutput


def scale_observation(t: np.array):
    return (t - np.min(t)) / (np.max(t) - np.min(t))


def scale_hidden_state(t: tf.Tensor):
    return (t - tf.reduce_min(t)) / (tf.reduce_max(t) - tf.reduce_min(t))


def build_policy_logits(policy_logits: tf.Tensor):
    return {Action(i): tf.squeeze(logit) for i, logit in enumerate(policy_logits[0])}


class Prediction(Model):
    def __init__(self, action_state_size: int, hidden_state_size: int):
        """
        p^k, v^k = f_0(s^k)
        :param action_state_size: size of action state
        """
        super(Prediction, self).__init__()
        neurons = 20
        self.inputs = InputLayer(input_shape=(hidden_state_size,), name="f_inputs")
        self.hidden = Dense(neurons, name="f_hidden", activation=tf.nn.relu)
        self.common = Dense(neurons, name="f_common", activation=tf.nn.relu)
        self.policy = Dense(action_state_size, name="f_policy")
        self.value = Dense(1, name="f_value")

    def call(self, hidden_state, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        :param hidden_state
        :return: NetworkOutput with policy logits and value
        """
        x = self.inputs(hidden_state)
        x = self.hidden(x)
        x = self.common(x)

        policy = self.policy(x)
        value = self.value(x)

        return policy, value


class Dynamics(Model):
    def __init__(self, hidden_state_size: int, enc_space_size: int):
        """
        r^k, s^k = g_0(s^(k-1), a^k)
        :param enc_space_size: size of hidden state
        """
        super(Dynamics, self).__init__()
        neurons = 20
        self.inputs = InputLayer(input_shape=(enc_space_size,), name="g_inputs")
        self.hidden = Dense(neurons, name="g_hidden", activation=tf.nn.relu)
        self.common = Dense(neurons, name="g_common", activation=tf.nn.relu)
        self.s_k = Dense(hidden_state_size, name="g_s_k")
        self.r_k = Dense(1, name="g_r_k")

    def call(self, encoded_space, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        :param **kwargs:
        :param **kwargs:
        :param encoded_space: hidden state concatenated with one_hot action
        :return: NetworkOutput with reward (r^k) and hidden state (s^k)
        """
        x = self.inputs(encoded_space)
        x = self.hidden(x)
        x = self.common(x)
        s_k = self.s_k(x)
        r_k = self.r_k(x)

        return s_k, r_k


class Representation(Model):
    def __init__(self, obs_space_size: int):
        """
        s^0 = h_0(o_1,...,o_t)
        :param obs_space_size
        """
        super(Representation, self).__init__()
        neurons = 20
        self.inputs = InputLayer(input_shape=(obs_space_size,), name="h_inputs")
        self.hidden = Dense(neurons,  name="h_hidden", activation=tf.nn.relu)
        self.common = Dense(neurons, name="h_common", activation=tf.nn.relu)
        self.s_0 = Dense(obs_space_size, name="h_s_0")

    def call(self, observation, **kwargs) -> tf.Tensor:
        """
        :param observation
        :return: state s0
        """
        x = self.inputs(observation)
        x = self.hidden(x)
        x = self.common(x)
        s_0 = self.s_0(x)
        return s_0


class Network(object):
    def __init__(self, config: MuZeroConfig):
        self.config = config
        self._training_steps = 0

        self.g_dynamics = Dynamics(config.state_space_size, config.action_space_size + config.state_space_size)
        self.f_prediction = Prediction(config.action_space_size, config.state_space_size)
        self.h_representation = Representation(config.state_space_size)

        self.f_prediction_path = Path('./checkpoints/muzero/Prediction')
        self.g_dynamics_path = Path('./checkpoints/muzero/Dynamics')
        self.h_representation_path = Path('./checkpoints/muzero/Representation')

    def save(self):
        Path.mkdir(self.f_prediction_path, parents=True, exist_ok=True)
        Path.mkdir(self.g_dynamics_path, parents=True, exist_ok=True)
        Path.mkdir(self.h_representation_path, parents=True, exist_ok=True)
        self.f_prediction.save(self.f_prediction_path, include_optimizer=False)
        self.g_dynamics.save(self.g_dynamics_path, include_optimizer=False)
        self.h_representation.save(self.h_representation_path, include_optimizer=False)

    def restore(self):
        if self.f_prediction_path.exists() and self.g_dynamics_path.exists() and self.h_representation_path.exists():
            self.f_prediction = tf.keras.models.load_model(self.f_prediction_path, compile=False)
            self.g_dynamics = tf.keras.models.load_model(self.g_dynamics_path, compile=False)
            self.h_representation = tf.keras.models.load_model(self.h_representation_path, compile=False)

    def initial_inference(self, observation: np.array) -> NetworkOutput:
        # representation + prediction function

        # representation
        observation = np.expand_dims(observation, axis=0)
        observation = scale_observation(observation)
        s_0 = self.h_representation(observation)

        # prediction
        policy, value = self.f_prediction(s_0)

        return NetworkOutput(
            value=tf.squeeze(value),
            reward=tf.constant(0.0),
            policy_logits=build_policy_logits(policy),
            hidden_state=s_0
        )

    def recurrent_inference(self, hidden_state, action: Action) -> NetworkOutput:
        # dynamics + prediction function
        # dynamics (encoded_state)
        hidden_state = scale_hidden_state(hidden_state)
        one_hot = tf.expand_dims(tf.one_hot(action.index, self.config.action_space_size), 0)
        encoded_state = tf.concat([hidden_state, one_hot], axis=1)

        s_k, r_k = self.g_dynamics(encoded_state)

        policy, value = self.f_prediction(s_k)

        return NetworkOutput(
            value=tf.squeeze(value),
            reward=tf.squeeze(r_k),
            policy_logits=build_policy_logits(policy),
            hidden_state=s_k
        )

    def get_weights(self) -> List:
        networks = self.get_networks()
        return [variable for network in networks for variable in network.trainable_weights]

    def get_variables(self):
        networks = self.get_networks()
        return [variable for network in networks for variable in network.trainable_variables]

    def get_networks(self) -> List:
        return [
            self.f_prediction,
            self.g_dynamics,
            self.h_representation
        ]

    def training_steps(self) -> int:
        # How many steps / batches the network has been trained for.
        return int(self._training_steps / self.config.batch_size)

    def training_steps_counter(self) -> int:
        return self._training_steps

    def increment_training_steps(self):
        self._training_steps += 1
