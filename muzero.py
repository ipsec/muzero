from config import MuZeroConfig
from core.core import SharedStorage, ReplayBuffer
from training.mcts import run_selfplay, train_network


def muzero(config: MuZeroConfig):
    storage = SharedStorage(config)
    replay_buffer = ReplayBuffer(config)

    for _ in range(config.num_actors):
        run_selfplay(config, storage, replay_buffer)

    train_network(config, storage, replay_buffer)

    return storage.latest_network()


def make_cartpole_config() -> MuZeroConfig:
    def visit_softmax_temperature():
        return 1.0

    return MuZeroConfig(
        observation_space_size=4,
        num_actors=1,
        lr_init=0.001,
        lr_decay_steps=5,
        action_space_size=2,
        max_moves=500,
        discount=0.99,
        dirichlet_alpha=0.25,
        num_simulations=11,  # Odd number perform better in eval mode
        batch_size=512,
        td_steps=10,
        visit_softmax_temperature_fn=visit_softmax_temperature)


if __name__ == '__main__':
    config = make_cartpole_config()
    muzero(config)
