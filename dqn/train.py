import datetime
import os
import matplotlib.pyplot as plt
import numpy as np

from gym import wrappers

DQN = "dqn"
MC_DROPOUT = "mc_dropout"
CONCRETE_DROPOUT = "concrete_dropout"
BAYES_BY_BACKPROP = "bayes_by_backprop"
MNF = "mnf"
ALLOWED_NETWORK_CONFIGS = {DQN, MC_DROPOUT, CONCRETE_DROPOUT, BAYES_BY_BACKPROP, MNF}
BAYES_NETWORK_CONFIGS = {MC_DROPOUT, CONCRETE_DROPOUT, BAYES_BY_BACKPROP, MNF}


def train_episode(env, train_net, target_net, config):
    rewards = 0
    reward_list = []
    losses = []
    kl_losses = []
    state = env.reset()
    algorithm = config["algorithm"]

    for step in range(1, config["step_limit"]+1):
        if config["env_render"] == True:
            env.render()

        # choose next action base on network
        if algorithm == DQN:
            action = train_net.get_action(state, epsilon=config["epsilon"])
        elif algorithm == BAYES_BY_BACKPROP:
            action = train_net.get_action(state, same_noise=True)
        elif algorithm == MNF:
            action = train_net.get_action(state, same_noise=True)
        elif algorithm == MC_DROPOUT:
            action = train_net.get_action(state, training=True)
        elif algorithm == CONCRETE_DROPOUT:
            action = train_net.get_action(state, training=True)

        prev_state = state  # store old observations
        state, reward, done, _ = env.step(action)  # execute action, observe reward and next state
        rewards = rewards + reward

        if step == (config["step_limit"]):
            done = True

        # store transitions
        exp = {'s': prev_state, 'a': action, 'r': reward, 's_next': state, 'end': done}
        train_net.add_experience(exp)

        if step % config["gradient_steps"] == 0:
            if algorithm in BAYES_NETWORK_CONFIGS:
                loss, kl_loss = train_net.train(target_net)
                kl_losses.append(kl_loss)
                losses.append(loss)
            else:
                loss = train_net.train(target_net)
                losses.append(loss)

        # copy weights every 'copy_steps' to target network
        if step % config["copy_steps"] == 0:
            target_net.copy_weights(train_net)

        if done:
            state = env.reset()
            reward_list.append(rewards)
            rewards = 0

    mean_loss = np.mean(losses)

    if algorithm in BAYES_NETWORK_CONFIGS:
        mean_kl = np.mean(kl_losses)
        return reward_list[0], step, mean_loss, mean_kl

    else:
        return reward_list[0], step, mean_loss


def test_policy(env, train_net, config, video=False):
    if video:
        env = wrappers.Monitor(env, os.path.join(os.getcwd(), "videos"), force=True)

    rewards = 0
    state = env.reset()
    algorithm = config["algorithm"]

    for step in range(config["step_limit"]):
        if config["env_render"] == True:
            env.render()

        # choose next action base on network
        if algorithm == DQN:
            action = train_net.get_action(state, epsilon=0)
        elif algorithm == BAYES_BY_BACKPROP:
            action = train_net.get_action(state, training=False)
        elif algorithm == MNF:
            action = train_net.get_action(state, training=False)
        elif algorithm == MC_DROPOUT:
            action = train_net.get_action(state, training=False)
        elif algorithm == CONCRETE_DROPOUT:
            action = train_net.get_action(state, training=False)

        state, reward, done, _ = env.step(action)
        rewards = rewards + reward

        if step == (config["step_limit"] - 1):
            done = True

        if done:
            break

    return rewards, step


def train_dqn(config, env, train_net, target_net, run_id):
    algorithm = config["algorithm"]
    if algorithm not in ALLOWED_NETWORK_CONFIGS:
        raise AssertionError(f"'algorithm' has to be one of {ALLOWED_NETWORK_CONFIGS} but is set to {algorithm}.")

    epsilon = config["epsilon"]
    n_epochs = config["epochs_num"]
    train_losses = np.empty(n_epochs)
    train_kl = np.empty(n_epochs)
    train_rewards = np.empty(n_epochs)

    test_rewards = [0]
    test_iterations = [0]
    mean_kl = 0
    total_steps = 0

    # initialize train and target net
    state = env.reset()
    _ = train_net.get_action(state)
    _ = target_net.get_action(state)
    if algorithm in {BAYES_BY_BACKPROP, MNF}:
        train_net.model.kl_div(same_noise=True)
        target_net.model.kl_div(same_noise=True)
    target_net.copy_weights(train_net)  # initialize with same weights

    for n in range(n_epochs):
        env.reset()  # initialize sequence

        if algorithm == DQN:
            epsilon = max(config["epsilon_min"], epsilon * config["epsilon_decay"])
            train_reward, steps, mean_loss = train_episode(env, train_net, target_net, config)

        elif algorithm == BAYES_BY_BACKPROP:
            if n > 0:
                train_net.model.reset_noise()
            train_reward, steps, mean_loss, mean_kl = train_episode(env, train_net, target_net, config)
            train_kl[n] = mean_kl

        elif algorithm == MNF:
            if n > 0:
                train_net.model.reset_noise()
            train_reward, steps, mean_loss, mean_kl = train_episode(env, train_net, target_net, config)
            train_kl[n] = mean_kl

        elif algorithm == MC_DROPOUT:
            train_reward, steps, mean_loss, mean_kl = train_episode(env, train_net, target_net, config)

        elif algorithm == CONCRETE_DROPOUT:
            train_reward, steps, mean_loss, mean_kl = train_episode(env, train_net, target_net, config)

        total_steps = total_steps + steps
        train_losses[n] = mean_loss
        train_rewards[n] = train_reward
        avg_train_rewards = train_rewards[max(0, n - 100):(n + 1)].mean()  # average reward of the last 100 episodes

        if n % config["test_episodes"] == 0:
            if n == 0:  # first episode is burn in phase
                total_reward = 0
                iterations = 0
            else:
                total_reward, iterations = test_policy(env, train_net, config)

            test_rewards.append(total_reward)
            test_iterations.append(total_steps)

            print(f"Epoch: {n}, reward: {total_reward}, loss: {mean_loss}, kl-loss: {mean_kl} iterations: {iterations}"
                  f", epsilon: {epsilon}, avg reward (last 100): {avg_train_rewards}")

    if config["plot_avg_reward"]:
        directory = f"results/plots/{algorithm}/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        plt.figure()
        filename = f"AccumulatedReward_{algorithm}_{str(run_id)}.pdf"
        plt.plot(test_iterations, test_rewards, linewidth=0.75)
        plt.xlabel("Iterations")
        plt.legend(["Accumulated reward"])
        plt.tight_layout()
        plt.savefig(os.path.join(directory, filename))
        plt.close()

        plt.figure()
        filename = f"Loss_{algorithm}_{str(run_id)}.pdf"
        plt.plot(range(config["epochs_num"]), train_losses, linewidth=0.75)
        plt.plot(range(config["epochs_num"]), train_kl, linewidth=0.75)
        plt.xlabel("Iterations")
        plt.legend(["Mean loss", "Mean kl-loss"])
        plt.tight_layout()
        plt.savefig(os.path.join(directory, filename))
        plt.close()

    if config["save"]:
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        save_dir = f"results/{config['env_name']}/{algorithm}/" + str(run_id) + '_' + current_time
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.savez(save_dir, test_rewards=test_rewards, test_iterations=test_iterations)
