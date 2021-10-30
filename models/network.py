from abc import ABC
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

from config import MuZeroConfig
from games.game import Action
from models import NetworkOutput
from utils import tf_support_to_scalar, tf_scalar_to_support, cast_to_tensor


def scale_gradient(tensor: tf.Tensor, scale: float) -> tf.Tensor:
    """
    Scale gradients for reverse differentiation proportional to the given scale.
    Does not influence the magnitude/ scale of the output from a given tensor (just the gradient).
    """
    return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)


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
        self.value = Dense(support_size * 2 + 1, name="f_value", activation=tf.nn.relu)

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
        self.s_k = Dense(hidden_state_size, name="g_s_k", activation=tf.nn.tanh)
        self.r_k = Dense(support_size * 2 + 1, name="g_r_k", activation=tf.nn.relu)

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
        self.s_0 = Dense(hidden_state_size, name="h_s_0", activation=tf.nn.tanh)

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

        # TODO: how initialize the weights the right way?
        self.f_prediction(np.atleast_2d(np.random.random(self.hidden_state_size)))
        self.g_dynamics(np.atleast_2d(np.random.random(encoded_size)))
        self.h_representation(np.atleast_2d(np.random.random(config.state_space_size)))

    def initial_inference(self, observations: np.array) -> NetworkOutput:
        # representation + prediction function

        # representation
        observations = observations[np.newaxis, ...]
        s_0 = self.h_representation(observations)

        # prediction
        policy, value = self.f_prediction(s_0)
        value = tf_support_to_scalar(value, self.config.support_size)

        return NetworkOutput(
            value=value,
            reward=0.,
            policy_logits=build_policy_logits(policy),
            hidden_state=s_0
        )

    def recurrent_inference(self, hidden_state: tf.Tensor, action: Action) -> NetworkOutput:
        # dynamics + prediction function
        # dynamics (encoded_state)
        one_hot = tf.one_hot([action.index], self.config.action_space_size)
        encoded_state = tf.concat([hidden_state, one_hot], axis=1)

        s_k, r_k = self.g_dynamics(encoded_state)

        policy, value = self.f_prediction(s_k)
        value = tf_support_to_scalar(value, self.config.support_size)
        r_k = tf_support_to_scalar(r_k, self.config.support_size)

        return NetworkOutput(
            value=value,
            reward=r_k,
            policy_logits=build_policy_logits(policy),
            hidden_state=s_k
        )

    def set_weights(self, weights: Dict):
        self.f_prediction.set_weights(weights.get('prediction'))
        self.g_dynamics.set_weights(weights.get('dynamics'))
        self.h_representation.set_weights(weights.get('representation'))

    def get_weights(self) -> Dict:
        return {
            'prediction': self.f_prediction.get_weights(),
            'dynamics': self.g_dynamics.get_weights(),
            'representation': self.h_representation.get_weights()
        }

    def get_networks(self) -> List:
        return [self.f_prediction, self.g_dynamics, self.h_representation]

    def get_variables(self):
        networks = [self.g_dynamics, self.f_prediction, self.h_representation]
        return [variables
                for variables_list in map(lambda n: n.weights, networks)
                for variables in variables_list]

    def training_steps(self) -> int:
        return self._training_steps

    def loss_function(self, batch) -> tf.Tensor:
        loss = 0.
        for image, actions, targets in batch:
            # Initial step, from the real observation.
            network_output = self.initial_inference(image)
            hidden_state = network_output.hidden_state
            predictions = [(1.0, network_output)]

            # Recurrent steps, from action and previous hidden state.
            for action in actions:
                network_output = self.recurrent_inference(hidden_state, action)
                hidden_state = network_output.hidden_state
                predictions.append((1.0 / len(actions), network_output))

                hidden_state = scale_gradient(hidden_state, 0.5)

            for k, (prediction, target) in enumerate(zip(predictions, targets)):
                gradient_scale, network_output = prediction
                target_value, target_reward, target_policy = target

                policy = list(network_output.policy_logits.values())

                if not target_policy:
                    continue

                l = tf.nn.softmax_cross_entropy_with_logits(logits=policy, labels=target_policy)
                l += scalar_loss(network_output.value, target_value, self.config.support_size)

                if k > 0:
                    l += scalar_loss(network_output.reward, target_reward, self.config.support_size)

                loss += scale_gradient(l, gradient_scale)
        loss /= len(batch)

        for weights in self.get_variables():
            loss += self.config.weight_decay * tf.nn.l2_loss(weights)

        return loss

    def update_training_steps(self):
        self._training_steps += 1

    def save(self):
        for model in self.get_networks():
            path = Path(f'./data/saved_model/{self.training_steps()}/{model.__class__.__name__}')
            Path.mkdir(path, parents=True, exist_ok=True)
            tf.saved_model.save(model, str(path.absolute()))


def scalar_loss(pred: float, target: float, support_size: int) -> tf.Tensor:
    pred = tf.expand_dims([cast_to_tensor(pred)], -1)
    target = tf.expand_dims([cast_to_tensor(target)], -1)
    pred = tf_scalar_to_support(pred, support_size)
    target = tf_scalar_to_support(target, support_size)
    return tf.reduce_sum(-target * tf.nn.log_softmax(pred, axis=1))
