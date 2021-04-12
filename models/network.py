from abc import ABC
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

from config import MuZeroConfig
from games.game import Action
from models import NetworkOutput
from utils import tf_support_to_scalar


def scale_hidden_state(t: tf.Tensor):
    return (t - tf.reduce_min(t)) / (tf.reduce_max(t) - tf.reduce_min(t))


def build_policy_logits(policy_logits: tf.Tensor):
    return {Action(i): logit for i, logit in enumerate(policy_logits[0])}


class Prediction(Model, ABC):
    def __init__(self, action_state_size: int, hidden_state_size: int, support_size: int):
        """
        p^k, v^k = f_0(s^k)
        :param action_state_size: size of action state
        """
        super(Prediction, self).__init__()
        neurons = 512
        self.inputs = Dense(hidden_state_size, name="f_inputs")
        self.hidden_policy = Dense(neurons, name="f_hidden_policy", activation=tf.nn.relu)
        self.hidden_value = Dense(neurons, name="f_hidden_value", activation=tf.nn.relu)
        self.policy = Dense(action_state_size, name="f_policy")
        self.value = Dense(2 * support_size + 1, name="f_value")

    @tf.function
    def call(self, hidden_state, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        :param hidden_state
        :return: NetworkOutput with policy logits and value
        """
        x = self.inputs(hidden_state)
        hidden_policy = self.hidden_policy(x)
        hidden_value = self.hidden_value(x)

        policy = self.policy(hidden_policy)
        value = self.value(hidden_value)

        return policy, value


class Dynamics(Model, ABC):
    def __init__(self, hidden_state_size: int, enc_space_size: int, support_size: int):
        """
        r^k, s^k = g_0(s^(k-1), a^k)
        :param enc_space_size: size of hidden state
        """
        super(Dynamics, self).__init__()
        neurons = 512
        self.inputs = Dense(enc_space_size, name="g_inputs")
        self.hidden_state = Dense(neurons, name="g_hidden_state", activation=tf.nn.relu)
        self.hidden_reward = Dense(neurons, name="g_hidden_reward", activation=tf.nn.relu)
        self.s_k = Dense(hidden_state_size, name="g_s_k", activation=tf.nn.relu)
        self.r_k = Dense(2 * support_size + 1, name="g_r_k")

    @tf.function
    def call(self, encoded_space, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        :param **kwargs:
        :param **kwargs:
        :param encoded_space: hidden state concatenated with one_hot action
        :return: NetworkOutput with reward (r^k) and hidden state (s^k)
        """
        x = self.inputs(encoded_space)
        hidden_state = self.hidden_state(x)
        hidden_reward = self.hidden_reward(x)
        s_k = self.s_k(hidden_state)
        r_k = self.r_k(hidden_reward)

        return s_k, r_k


class Representation(Model, ABC):
    def __init__(self, obs_space_size: int, hidden_state_size: int):
        """
        s^0 = h_0(o_1,...,o_t)
        :param obs_space_size
        """
        super(Representation, self).__init__()
        neurons = 512
        self.inputs = Dense(obs_space_size, name="h_inputs")
        self.hidden = Dense(neurons, name="h_hidden", activation=tf.nn.relu)
        self.s_0 = Dense(hidden_state_size, name="h_s_0", activation=tf.nn.relu)

    @tf.function
    def call(self, observation, **kwargs) -> tf.Tensor:
        """
        :param observation
        :return: state s0
        """
        x = self.inputs(observation)
        x = self.hidden(x)
        s_0 = self.s_0(x)
        return s_0


class Network(object):
    def __init__(self, config: MuZeroConfig):
        self.config = config
        self._training_steps = 0

        self.hidden_state_size = 50

        encoded_size = config.action_space_size + self.hidden_state_size

        self.f_prediction = Prediction(config.action_space_size, self.hidden_state_size, self.config.support_size)
        self.g_dynamics = Dynamics(self.hidden_state_size, encoded_size, self.config.support_size)
        self.h_representation = Representation(config.state_space_size, self.hidden_state_size)

        self.f_prediction_path = Path('./checkpoints/muzero')
        self.g_dynamics_path = Path('./checkpoints/muzero')
        self.h_representation_path = Path('./checkpoints/muzero')

    def save(self, step: int):
        # Saving step folder
        prediction_step_path = self.f_prediction_path.joinpath(str(step)).joinpath("Prediction")
        dynamics_step_path = self.g_dynamics_path.joinpath(str(step)).joinpath("Dynamics")
        representation_step_path = self.h_representation_path.joinpath(str(step)).joinpath("Representation")
        Path.mkdir(prediction_step_path, parents=True, exist_ok=True)
        Path.mkdir(dynamics_step_path, parents=True, exist_ok=True)
        Path.mkdir(representation_step_path, parents=True, exist_ok=True)
        try:
            self.f_prediction.save(prediction_step_path, include_optimizer=False)
            self.g_dynamics.save(dynamics_step_path, include_optimizer=False)
            self.h_representation.save(representation_step_path, include_optimizer=False)
        except Exception:
            pass

    def save_latest(self):
        # Saving latest folder
        prediction_latest_path = self.f_prediction_path.joinpath('latest').joinpath("Prediction")
        dynamics_latest_path = self.g_dynamics_path.joinpath('latest').joinpath("Dynamics")
        representation_latest_path = self.h_representation_path.joinpath('latest').joinpath("Representation")
        Path.mkdir(prediction_latest_path, parents=True, exist_ok=True)
        Path.mkdir(dynamics_latest_path, parents=True, exist_ok=True)
        Path.mkdir(representation_latest_path, parents=True, exist_ok=True)
        try:
            self.f_prediction.save(prediction_latest_path, include_optimizer=False)
            self.g_dynamics.save(dynamics_latest_path, include_optimizer=False)
            self.h_representation.save(representation_latest_path, include_optimizer=False)
        except Exception:
            pass

    def restore(self):
        prediction_latest_path = self.f_prediction_path.joinpath('latest').joinpath("Prediction")
        dynamics_latest_path = self.g_dynamics_path.joinpath('latest').joinpath("Dynamics")
        representation_latest_path = self.h_representation_path.joinpath('latest').joinpath("Representation")

        if prediction_latest_path.exists() and dynamics_latest_path.exists() and representation_latest_path.exists():
            try:
                self.f_prediction = load_model(prediction_latest_path, compile=False)
                self.g_dynamics = load_model(dynamics_latest_path, compile=False)
                self.h_representation = load_model(representation_latest_path, compile=False)
            except Exception:
                pass

    def initial_inference(self, observation: np.array, training: bool = False) -> NetworkOutput:
        # representation + prediction function

        # representation
        # observation = tf.expand_dims(observation, axis=0)
        # observation = scale_observation(observation)
        s_0 = self.h_representation(observation, training=training)

        s_0 = scale_hidden_state(s_0)

        # prediction
        policy, value = self.f_prediction(s_0, training=training)
        value = tf_support_to_scalar(value, self.config.support_size)

        return NetworkOutput(
            value=tf.squeeze(value),
            reward=tf.constant(0.0),
            policy_logits=build_policy_logits(policy),
            hidden_state=s_0
        )

    def recurrent_inference(self, hidden_state: tf.Tensor, action: Action, training: bool = False) -> NetworkOutput:
        # dynamics + prediction function
        # dynamics (encoded_state)
        one_hot = tf.one_hot([action.index], self.config.action_space_size)
        encoded_state = tf.concat([hidden_state, one_hot], axis=1)

        s_k, r_k = self.g_dynamics(encoded_state, training=training)

        s_k = scale_hidden_state(s_k)

        policy, value = self.f_prediction(s_k, training=training)
        value = tf_support_to_scalar(value, self.config.support_size)
        r_k = tf_support_to_scalar(r_k, self.config.support_size)

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
        return [self.f_prediction, self.g_dynamics, self.h_representation]

    def training_steps(self) -> int:
        # How many steps / batches the network has been trained for.
        return int(self._training_steps / self.config.batch_size)

    def training_steps_counter(self) -> int:
        return self._training_steps

    def increment_training_steps(self):
        self._training_steps += 1
