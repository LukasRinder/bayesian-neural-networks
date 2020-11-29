# Normalizing Flows

This repository investigates recent variational Bayesian inference approaches for uncertainty estimation. The approaches
are evaluated and visualized on simple regression tasks. Furthermore, the uncertainty estimates from the variational 
Bayesian neural networks are used to perform approximate Thompson sampling within a deep Q-network (DQN) for efficient 
exploration. The approaches are compared against each other and against the well known epsilon-greedy strategy.

Currently, following variational Bayesian neural networks are implemented:
 
- Bayes by Backprop [1]
- Multiplicative Normalizing Flows (MNF) [2]
- Dropout as a Bayesian Approximation [3]
- Concrete Dropout [4]

Touati et al. [5] describe how to augment DQNs with multiplicative normalizing flows for an efficient 
exploration-exploitation strategy.

The repository is structured in the following way:
- [bayes](/bayes) contains implementations of Bayes By Backprop, MNF, and Concrete Dropout layers. Monte Carlo 
dropout utilizes the standard Tensorflow dropout layer.
- [data](/data) contains two regression data sets mentioned in [6] and [7] used to visualize the uncertainty estimates.
- [dqn](/dqn) includes the DQN implementations utilizing the respective variational Bayesian neural networks.
- [envs](/envs) includes an implementation of a N-chain gym environment and environment utility functions.
- [normalizingflows](/normalizingflows) contains normalizing flows for the use in Multiplicative Normalizing Flows.
- [plots](/plots) contains some example visualizations.

Training functions are located at the root of the repository.

Below we show some example uncertainty estimates on the regression task mentioned in [6]. Additionally, we show the
average accumulated reward over 5 runs on the OpenAi gym envionments CartPole and MountainCar.

- Aleatoric (data) uncertainty vs. epistemic (knowledge) uncertainty predicted by MC Dropout with two network heads:

<img src="plots/MCDropout_heteroscedastic.pdf" width="500" height="200" />


- Network utilizing 3 MNF dense layers:

<img src="plots/MNF_all_layers.pdf" width="500" height="200" />


- Network utilizing 2 regular dense layers and 1 MNF dense layers:

<img src="plots/MNF_last_layers.pdf" width="500" height="200" />


- Average accumulated reward over 5 runs on the OpenAI gym CartPole task:

<img src="plots/avg_acc_reward_cartpole.pdf" width="500" height="200" />


- Average accumulated reward over 5 runs on the OpenAI gym MountainCar task:

<img src="plots/avg_acc_reward_mountaincar.pdf" width="500" height="200" />


This work was done during the Advanced Deep Learning for Robotics course at TUM in cooperation with the German Aerospace 
Center (DLR).
In case of any questions, feel free to reach out to us.

Jan Rüttinger, jan.ruettinger@tum.de

Lukas Rinder, lukas.rinder@tum.de


### References

[1] C. Blundell, J. Cornebise, K. Kavukcuoglu, and D. Wierstra, “Weight uncertainty in neural networks,” 32nd Int. Conf. Mach. Learn. ICML 2015, vol. 2, pp. 1613–1622, 2015.

[2] C. Louizos and M. Welling, “Multiplicative normalizing flows for variational Bayesian neural networks,” 34th Int. Conf. Mach. Learn. ICML 2017, vol. 5, pp. 3480–3489, 2017.

[3] Y. Gal and Z. Ghahramani, “Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning,” 33rd Int. Conf. Mach. Learn. ICML 2016, vol. 3, pp. 1651–1660, Jun. 2015.

[4] Y. Gal, J. Hron, and A. Kendall, “Concrete dropout,” in Advances in Neural Information Processing Systems, 2017, vol. 2017-Decem, pp. 3582–3591.

[5] A. Touati, H. Satija, J. Romoff, J. Pineau, and P. Vincent, “Randomized value functions via multiplicative normalizing flows,” 35th Conf. Uncertain. Artif. Intell. UAI 2019, 2019.

[6] I. Osband, “Risk versus uncertainty in deep learning: Bayes, bootstrap and the dangers of dropout.,” NIPS Work. Bayesian Deep Learn., vol. 192, 2016.

[7] J. M. Hernández-Lobato and R. P. Adams, “Probabilistic backpropagation for scalable learning of Bayesian neural networks,” 32nd Int. Conf. Mach. Learn. ICML 2015, vol. 3, pp. 1861–1869, 2015.
