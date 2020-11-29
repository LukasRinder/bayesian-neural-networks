import gym
import tensorflow as tf

from envs.env_utils import WrapFrameSkip
from dqn.Concrete_Dropout_DQN import DQN
from dqn.train import train_dqn

# config cart pole
CONFIG_CARTPOLE = {
    "env_name": "CartPole-v1",
    "algorithm": "concrete_dropout",
    "seed": [210, 142, 531, 461, 314],
    "runs": 1,  # perform e.g. 5 runs
    "env_render": True,
    "alpha": 1,
    "skip_frame_num": 0,
    "epochs_num": 50,
    "hidden_units": "100,100",  # 400, 400
    "gradient_update_gamma":  0.9,
    "batch_size": 64,
    "learning_rate_init": 1e-3,
    "experiences_max": 5000,
    "experiences_min": 200,
    "epsilon_min": None,
    "epsilon": None,
    "epsilon_decay": None,
    "copy_steps": 25,
    "gradient_steps": 1,
    "step_limit": 200,
    "test_episodes": 5,  # perform a test episode after 'test episode' many train epochs
    "plot_avg_reward": True,
    "save": False,  # saves a npz-file with the data of the runs
}

# config mountain car
CONFIG_MOUNTAINCAR = {
    "env_name": "MountainCar-v0",
    "algorithm": "concrete_dropout",
    "seed": [210, 142, 531, 461, 314],
    "runs": 1,  # perform e.g. 5 runs
    "env_render": True,
    "alpha": 1,
    "skip_frame_num": 4,
    "epochs_num": 100,
    "hidden_units": "200,200,200,200",
    "gradient_update_gamma":  0.9,
    "batch_size": 64,
    "learning_rate_init": 1e-3,
    "experiences_max": 5000,
    "experiences_min": 200,
    "epsilon_min": None,
    "epsilon": None,
    "epsilon_decay": None,
    "copy_steps": 25,
    "gradient_steps": 1,
    "step_limit": 500,
    "test_episodes": 10,  # perform a test episode after 'test episode' many train epochs
    "plot_avg_reward": True,
    "save": False,  # saves a npz-file with the data of the runs
}

config = CONFIG_CARTPOLE  # switch between cart pole and mountain car

config_static = {
    "learning_rate": tf.keras.optimizers.schedules.PolynomialDecay(config["learning_rate_init"],
                                                                   config["epochs_num"]*config["step_limit"], 1e-5,
                                                                   power=0.5)
}

# Setup environment
env = gym.make(config["env_name"]).env  # remove 200 step limit

if config["skip_frame_num"] > 0:    # optional: skip frames to ease training in MountainCar
    env = WrapFrameSkip(env, frameskip=config["skip_frame_num"])

num_states = len(env.observation_space.sample())
num_actions = env.action_space.n
print(f"Number of available actions: {num_actions}")
print(f"Available action values (force on the cart in N): {env.action_space}")

hidden_units = []
for i in config["hidden_units"].split(","):
    hidden_units.append(int(i))

print(f"GPU available: {tf.test.is_gpu_available()}")

for run_id in (range(config["runs"])):
    tf.random.set_seed(config["seed"][run_id])

    # initialize train (action-value function) and target network (target action-value function)
    train_net = DQN(num_states=num_states, num_actions=num_actions, hidden_units=hidden_units,
                    gamma=config["gradient_update_gamma"], max_experiences=config["experiences_max"],
                    min_experiences=config["experiences_min"], batch_size=config["batch_size"],
                    lr=config_static["learning_rate"])
    target_net = DQN(num_states=num_states, num_actions=num_actions, hidden_units=hidden_units,
                     gamma=config["gradient_update_gamma"], max_experiences=config["experiences_max"],
                     min_experiences=config["experiences_min"], batch_size=config["batch_size"],
                     lr=config_static["learning_rate"])

    train_dqn(config, env, train_net, target_net, run_id)
