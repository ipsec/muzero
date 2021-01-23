from typing import NamedTuple, Dict, List

import numpy as np

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense


class NetworkOutput(NamedTuple):
    value: float
    reward: float
    policy_logits: Dict[Action, float]
    hidden_state: List[float]

    @staticmethod
    def build_policy_logits(policy_logits):
        return {Action(i): logit for i, logit in enumerate(policy_logits[0])}


class RepresentationOutput(NamedTuple):
    state_zero: List[Tensor]


class DynamicsOutput(NamedTuple):
    state_k: List[Tensor]
    reward_k: Tensor


class PredictionOutput(NamedTuple):
    policy_logits: List[Tensor]
    value: Tensor


class Representation(Model):
    def __init__(self, observation_space_size: int):
        """
        s^0 = h_0(o_1,...,o_t)
        :param observation_space_size
        """
        super(Representation, self).__init__()
        neurons = 128
        self.inputs = Dense(neurons, activation='relu')
        self.hidden = Dense(neurons, activation='relu')
        self.common = Dense(neurons, activation='relu')
        self.s0 = Dense(observation_space_size, activation='relu')

    @tf.function
    def call(self, observation, **kwargs) -> RepresentationOutput:
        """
        :param observation
        :return: state s0
        """
        x = self.inputs(observation)
        x = self.hidden(x)
        x = self.common(x)
        s_0 = self.s0(x)
        return RepresentationOutput(state_zero=s_0)


class Dynamics(Model):
    def __init__(self, observation_space_size: int):
        """
        r^k, s^k = g_0(s^(k-1), a^k)
        :param observation_space_size: size of observation_space_size
        """
        super(Dynamics, self).__init__()
        neurons = 128
        self.inputs = Dense(neurons, activation='relu')
        self.hidden = Dense(neurons, activation='relu')
        self.common = Dense(neurons, activation='relu')
        self.s_k = Dense(observation_space_size, activation='relu')
        self.r_k = Dense(1, activation='relu')

    @tf.function
    def call(self, encoded_space, **kwargs) -> DynamicsOutput:
        """
        :param encoded_space: hidden state concatenated with one_hot action
        :return: DynamicsOutput with reward (r^k) and hidden state (s^k)
        """
        x = self.inputs(encoded_space)
        x = self.hidden(x)
        x = self.common(x)
        s_k = self.s_k(x)
        r_k = self.r_k(x)
        return DynamicsOutput(reward_k=r_k, state_k=s_k)


class Prediction(Model):
    def __init__(self, action_space_size: int):
        """
        p^k, v^k = f_0(s^k)
        :param action_space_size
        """
        super(Prediction, self).__init__()
        neurons = 128
        self.inputs = Dense(neurons, activation='relu')
        self.hidden = Dense(neurons, activation='relu')
        self.common = Dense(neurons, activation='relu')
        self.policy = Dense(action_space_size, activation='relu')
        self.value = Dense(1, activation='relu')

    @tf.function
    def call(self, hidden_state, **kwargs) -> PredictionOutput:
        """
        :param hidden_state
        :return:PredictionOutput with policy_logits and value
        """
        x = self.inputs(hidden_state)
        x = self.hidden(x)
        x = self.common(x)
        policy = self.policy(x)
        value = self.value(x)
        return PredictionOutput(policy_logits=policy, value=value)


class Network(object):

    def __init__(self, observation_space_size: int, action_space_size: int):
        self.observation_space_size = observation_space_size
        self.action_space_size = action_space_size
        self.h = Representation(self.observation_space_size)
        self.g = Dynamics(self.observation_space_size)
        self.f = Prediction(self.action_space_size)
        self.training_steps = 0

    def initial_inference(self, observation) -> NetworkOutput:
        """
        Esse metodo é chamado somente no nó raiz da árvore, por isso a reward é 0.
        :param observation
        :return: NetworkOutput
        """

        h_output = self.h(np.atleast_2d(observation))
        f_output = self.f(h_output.state_zero)

        # reward equal to 0 for consistency
        r_0 = 0.

        policy_logits = NetworkOutput.build_policy_logits(f_output.policy_logits)
        return NetworkOutput(
            value=f_output.value,
            reward=r_0,
            policy_logits=policy_logits,
            hidden_state=h_output.state_zero)

    def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
        encoded_space = self.get_encoded_space(hidden_state, action)
        g_output = self.g(encoded_space)
        f_output = self.f(g_output.state_k)

        policy_logits = NetworkOutput.build_policy_logits(f_output.policy_logits)

        return NetworkOutput(
            value=f_output.value,
            reward=g_output.reward_k,
            policy_logits=policy_logits,
            hidden_state=g_output.state_k)

    def get_encoded_space(self, state, action) -> Tensor:
        """
        A entrada para a função g é o resultado do one_hot concatenado ao hidden_state
        conforme 'Observation and action encoding' do documento
        :param state: hidden_state
        :param action:
        :return:
        """
        one_hot = np.atleast_2d(tf.one_hot(action.index, self.action_space_size))
        return tf.concat([state, one_hot], axis=1)

    def get_weights(self) -> list:
        return [self.h.get_weights(), self.g.get_weights(), self.f.get_weights()]

    def get_variables(self) -> list:
        networks = (self.h, self.g, self.f)
        return [variables
                for variables_list in map(lambda n: n.weights, networks)
                for variables in variables_list]
