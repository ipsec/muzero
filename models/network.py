from abc import ABC
from pathlib import Path
from typing import Callable, List

import tensorflow as tf
from tensorflow.keras.initializers import Zeros, RandomUniform
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

from config import MuZeroConfig
from games.game import Action
from models import NetworkOutput
from utils import tf_support_to_scalar


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
        reward_initializer = Zeros()
        self.s_inputs = Dense(enc_space_size,
                              input_shape=(enc_space_size,),
                              name="g_s_input")
        self.s_hidden = Dense(neurons,
                              name="g_s_hidden")
        self.s_k = Dense(hidden_state_size,
                         activation=tf.nn.tanh,
                         name="g_s_k")

        self.r_inputs = Dense(enc_space_size,
                              input_shape=(enc_space_size,),
                              kernel_initializer=reward_initializer,
                              name="g_r_input")
        self.r_hidden = Dense(neurons,
                              kernel_initializer=reward_initializer,
                              name="g_r_hidden")
        self.r_k = Dense(41,
                         kernel_initializer=reward_initializer,
                         name="g_r_k")

    @tf.function
    def call(self, encoded_space, **kwargs):
        """
        :param **kwargs:
        :param encoded_space: hidden state concatenated with one_hot action
        :return: NetworkOutput with reward (r^k) and hidden state (s^k)
        """
        s = self.s_inputs(encoded_space)
        s = self.s_hidden(s)
        s_k = self.s_k(s)

        r = self.r_inputs(encoded_space)
        r = self.r_hidden(r)
        r_k = self.r_k(r)

        return s_k, r_k


class Prediction(Model, ABC):
    def __init__(self, action_state_size: int, hidden_state_size: int):
        """
        p^k, v^k = f_0(s^k)
        :param action_state_size: size of action state
        """
        super(Prediction, self).__init__()
        neurons = 20
        policy_initializer = RandomUniform(minval=0., maxval=1.)
        value_initializer = Zeros()
        self.p_inputs = Dense(hidden_state_size,
                              input_shape=(hidden_state_size,),
                              kernel_initializer=policy_initializer,
                              name="f_p_inputs")
        self.p_hidden = Dense(neurons,
                              kernel_initializer=policy_initializer,
                              name="f_p_hidden")
        self.policy = Dense(action_state_size,
                            kernel_initializer=policy_initializer,
                            name="f_policy")

        self.v_inputs = Dense(hidden_state_size,
                              input_shape=(hidden_state_size,),
                              kernel_initializer=value_initializer,
                              name="f_v_inputs")
        self.v_hidden = Dense(neurons,
                              kernel_initializer=value_initializer,
                              name="f_v_hidden")
        self.value = Dense(41,
                           kernel_initializer=value_initializer,
                           name="f_value")

    @tf.function
    def call(self, hidden_state, **kwargs):
        """
        :param hidden_state
        :return: NetworkOutput with policy logits and value
        """
        p = self.p_inputs(hidden_state)
        p = self.p_hidden(p)
        policy = self.policy(p)

        v = self.v_inputs(hidden_state)
        v = self.v_hidden(v)
        value = self.value(v)

        return policy, value


class Representation(Model, ABC):
    def __init__(self, obs_space_size: int):
        """
        s^0 = h_0(o_1,...,o_t)
        :param obs_space_size
        """
        super(Representation, self).__init__()
        neurons = 20
        self.inputs = Dense(obs_space_size,
                            input_shape=(obs_space_size,),
                            name="h_inputs")
        self.hidden = Dense(neurons,
                            name="h_hidden1")
        self.s0 = Dense(obs_space_size,
                        activation=tf.nn.tanh,
                        name="h_s0")

    @tf.function
    def call(self, observation, **kwargs):
        """
        :param observation
        :return: state s0
        """
        x = self.inputs(observation)
        x = self.hidden(x)
        s_0 = self.s0(x)
        return s_0


class Network(object):
    def __init__(self, config: MuZeroConfig):
        self.config = config
        self.g_dynamics = Dynamics(config.state_space_size, config.action_space_size + config.state_space_size)
        self.f_prediction = Prediction(config.action_space_size, config.state_space_size)
        self.h_representation = Representation(config.state_space_size)
        self._training_steps = 0

        self.g_dynamics_checkpoint = tf.train.Checkpoint(model=self.g_dynamics)
        self.f_prediction_checkpoint = tf.train.Checkpoint(model=self.f_prediction)
        self.h_representation_checkpoint = tf.train.Checkpoint(model=self.h_representation)

        self.g_dynamics_checkpoint_path = './checkpoints/muzero/Dynamics'
        Path.mkdir(Path(self.g_dynamics_checkpoint_path), parents=True, exist_ok=True)
        self.manager_dynamics = tf.train.CheckpointManager(self.g_dynamics_checkpoint,
                                                           directory=self.g_dynamics_checkpoint_path,
                                                           max_to_keep=5)

        self.f_prediction_checkpoint_path = './checkpoints/muzero/Prediction'
        Path.mkdir(Path(self.f_prediction_checkpoint_path), parents=True, exist_ok=True)
        self.manager_prediction = tf.train.CheckpointManager(self.f_prediction_checkpoint,
                                                             directory=self.f_prediction_checkpoint_path,
                                                             max_to_keep=5)

        self.h_representation_checkpoint_path = './checkpoints/muzero/Representation'
        Path.mkdir(Path(self.h_representation_checkpoint_path), parents=True, exist_ok=True)
        self.manager_representation = tf.train.CheckpointManager(self.h_representation_checkpoint,
                                                                 directory=self.h_representation_checkpoint_path,
                                                                 max_to_keep=5)

    def save_checkpoint(self):
        self.manager_dynamics.save()
        self.manager_prediction.save()
        self.manager_representation.save()

    def restore_checkpoint(self):
        self.g_dynamics_checkpoint.restore(self.manager_dynamics.latest_checkpoint)
        self.f_prediction_checkpoint.restore(self.manager_prediction.latest_checkpoint)
        self.h_representation_checkpoint.restore(self.manager_representation.latest_checkpoint)

    def prepare_observation(self, observation: tf.Tensor) -> tf.Tensor:
        observation = tf.expand_dims(observation, 0)
        # observation = scale(observation)
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

        v = tf_support_to_scalar(v, 20)

        return NetworkOutput(
            value=float(v.numpy()),
            reward=0.0,
            policy_logits=build_policy_logits(policy_logits=p),
            hidden_state=s_0,
        )

    @tf.function
    def encode_state(self, hidden_state: tf.Tensor, action: int, action_space_size: int) -> tf.Tensor:
        one_hot = tf.expand_dims(tf.one_hot(action, action_space_size), 0)
        encoded_state = tf.concat([hidden_state, one_hot], axis=1)
        return encoded_state

    def recurrent_inference(self, hidden_state, action: Action) -> NetworkOutput:
        # dynamics + prediction function
        # dynamics (encoded_state)
        encoded_state = self.encode_state(hidden_state, action.index, self.config.action_space_size)

        s_k, r_k = self.g_dynamics(encoded_state)
        # s_k = scale(s_k)

        r_k = tf_support_to_scalar(r_k, 20)

        # prediction
        p, v = self.f_prediction(s_k)
        v = tf_support_to_scalar(v, 20)

        return NetworkOutput(
            value=float(v.numpy()),
            reward=float(r_k.numpy()),
            policy_logits=build_policy_logits(policy_logits=p),
            hidden_state=s_k
        )

    def get_weights(self) -> List:
        networks = [self.g_dynamics, self.f_prediction, self.h_representation]
        return [variables
                for variables_list in map(lambda n: n.weights, networks)
                for variables in variables_list]

    def cb_get_variables(self) -> Callable:
        """Return a callback that return the trainable variables of the network."""

        def get_variables():
            networks = [self.g_dynamics, self.f_prediction, self.h_representation]
            return [variables
                    for variables_list in map(lambda n: n.weights, networks)
                    for variables in variables_list]

        return get_variables

    def get_networks(self) -> List:
        return [self.g_dynamics, self.f_prediction, self.h_representation]

    def training_steps(self) -> int:
        # How many steps / batches the network has been trained for.
        return int(self._training_steps / self.config.batch_size)

    def training_steps_counter(self) -> int:
        return self._training_steps

    def increment_training_steps(self):
        self._training_steps += 1

    def get_variables_by_network(self):
        return [[x.trainable_variables] for x in [self.g_dynamics, self.f_prediction, self.h_representation]]

    def get_variables(self):
        return [x.trainable_variables for x in [self.g_dynamics, self.f_prediction, self.h_representation]]
