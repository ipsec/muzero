from abc import ABC
from pathlib import Path
from typing import List

import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.models import Model

from config import MuZeroConfig
from games.game import Action
from models import NetworkOutput


def scale(t: tf.Tensor):
    return (t - tf.reduce_min(t)) / (tf.reduce_max(t) - tf.reduce_min(t))


def build_policy_logits(policy_logits):
    return {Action(i): logit for i, logit in enumerate(policy_logits[0])}


class Dynamics(Model, ABC):
    def __init__(self, hidden_state_size: int, enc_space_size: int):
        """
        r^k, s^k = g_0(s^(k-1), a^k)
        :param enc_space_size: size of hidden state
        """
        super(Dynamics, self).__init__()
        neurons = 20
        self.inputs = InputLayer(input_shape=(enc_space_size,), name="g_inputs")
        self.hidden = Dense(neurons, activation=tf.nn.relu, name="g_hidden")
        self.common = Dense(neurons, activation=tf.nn.relu, name="g_common")
        self.s_k = Dense(hidden_state_size, activation=tf.nn.tanh, name="g_s_k")
        self.r_k = Dense(1, name="g_r_k")

    @tf.function
    def call(self, encoded_space, **kwargs):
        """
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


class Prediction(Model, ABC):
    def __init__(self, action_state_size: int, hidden_state_size: int):
        """
        p^k, v^k = f_0(s^k)
        :param action_state_size: size of action state
        """
        super(Prediction, self).__init__()
        neurons = 20
        self.inputs = InputLayer(input_shape=(hidden_state_size,), name="f_inputs")
        self.hidden = Dense(neurons, activation=tf.nn.relu, name="f_hidden")
        self.common = Dense(neurons, activation=tf.nn.relu, name="f_common")
        self.policy = Dense(action_state_size, name="f_policy")
        self.value = Dense(1, name="f_value")

    @tf.function
    def call(self, hidden_state, **kwargs):
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


class Representation(Model, ABC):
    def __init__(self, obs_space_size: int):
        """
        s^0 = h_0(o_1,...,o_t)
        :param obs_space_size
        """
        super(Representation, self).__init__()
        neurons = 20
        self.inputs = InputLayer(input_shape=(obs_space_size,), name="h_inputs")
        self.hidden = Dense(neurons, activation=tf.nn.relu, name="h_hidden")
        self.common = Dense(neurons, activation=tf.nn.relu, name="h_common")
        self.s0 = Dense(obs_space_size, activation=tf.nn.tanh, name="h_s0")

    @tf.function
    def call(self, observation, **kwargs):
        """
        :param observation
        :return: state s0
        """
        x = self.inputs(observation)
        x = self.hidden(x)
        x = self.common(x)
        s_0 = self.s0(x)
        return s_0


class Network(object):
    def __init__(self, config: MuZeroConfig):
        self.config = config
        self.g_dynamics = Dynamics(config.state_space_size, config.action_space_size + config.state_space_size)
        self.f_prediction = Prediction(config.action_space_size, config.state_space_size)
        self.h_representation = Representation(config.state_space_size)
        self._training_steps = 0

        self.f_prediction_path = './checkpoints/muzero/Prediction'
        self.g_dynamics_path = './checkpoints/muzero/Dynamics'
        self.h_representation_path = './checkpoints/muzero/Representation'
        Path.mkdir(Path(self.f_prediction_path), parents=True, exist_ok=True)
        Path.mkdir(Path(self.g_dynamics_path), parents=True, exist_ok=True)
        Path.mkdir(Path(self.h_representation_path), parents=True, exist_ok=True)

    def save_checkpoint(self):
        self.f_prediction.save(self.f_prediction_path)
        self.g_dynamics.save(self.g_dynamics_path)
        self.h_representation.save(self.h_representation_path)

    def restore(self):
        try:
            self.f_prediction = tf.keras.models.load_model(self.f_prediction_path, compile=False)
        except OSError:
            """
            If model not trained yet, the saved model doesn't exists yet, then just pass
            """
            pass

    def prepare_observation(self, observation: tf.Tensor) -> tf.Tensor:
        observation = tf.expand_dims(observation, 0)
        observation = scale(observation)
        observation = tf.cast(observation, dtype=tf.float32)
        return observation

    def initial_inference(self, observation) -> NetworkOutput:
        # representation + prediction function
        # representation
        observation = self.prepare_observation(observation)

        s_0 = self.h_representation(observation)
        # s_0 = scale(s_0)

        # prediction
        p, v = self.f_prediction(s_0)

        # v = tf_support_to_scalar(v, 20)

        return NetworkOutput(
            value=float(v.numpy()),
            reward=0.0,
            policy_logits=build_policy_logits(policy_logits=p),
            hidden_state=s_0,
        )

    def encode_state(self, hidden_state: tf.Tensor, action: int, action_space_size: int) -> tf.Tensor:
        one_hot = tf.expand_dims(tf.one_hot(action, action_space_size), 0)
        encoded_state = tf.concat([hidden_state, one_hot], axis=1)
        # scale(encoded_state)
        return encoded_state

    def recurrent_inference(self, hidden_state, action: Action) -> NetworkOutput:
        # dynamics + prediction function
        # dynamics (encoded_state)
        encoded_state = self.encode_state(hidden_state, action.index, self.config.action_space_size)

        s_k, r_k = self.g_dynamics(encoded_state)
        # s_k = scale(s_k)

        # r_k = tf_support_to_scalar(r_k, 20)

        # prediction
        p, v = self.f_prediction(s_k)
        # v = tf_support_to_scalar(v, 20)

        return NetworkOutput(
            value=float(v.numpy()),
            reward=float(r_k.numpy()),
            policy_logits=build_policy_logits(policy_logits=p),
            hidden_state=s_k
        )

    def get_weigths(self) -> List:
        networks = (self.f_prediction, self.g_dynamics, self.h_representation)
        return [variable for network in networks for variable in network.weights]

    def get_variables(self):
        networks = (self.f_prediction, self.g_dynamics, self.h_representation)
        return [variable for network in networks for variable in network.trainable_variables]

    def get_networks(self) -> List:
        return [self.g_dynamics, self.f_prediction, self.h_representation]

    def training_steps(self) -> int:
        # How many steps / batches the network has been trained for.
        return int(self._training_steps / self.config.batch_size)

    def training_steps_counter(self) -> int:
        return self._training_steps

    def training_steps_set(self, value: int):
        self._training_steps = value

    def increment_training_steps(self):
        self._training_steps += 1
