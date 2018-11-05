# Notes-RL
A notes collection of important papers with the focus on reinforcement learning. Each note contains a compact summary of key idea in the paper, that is which problem the paper tries to solve and how they solve it. Second part is the detailed mathematical derivations.

## Structure of notes:
- Compact summary of key idea: which problem the paper tries to solve ? How they solve it ?
- Mathematical derivations of useful tools presented in the paper.
- File type:
  - Non-math (easy and quick): Markdown and figures
  - Math-based: PDF+LaTeX

## Main focus:
Based on my own research interests, we have following main focuses
- Model-based RL
- Optimization
- Information theory

# Table of contents
- [Textbook](#textbook)
- [Policy Gradients](#policy-gradients)
- [DQN](#dqn)
- [Model-based RL & Planning](#model-based-rl--planning)
- [RL Theory](#rl-theory)
- [Misc RL](#misc-rl)
- [Optimization & Variational Inference](#optimization--variational-inference)
- [Misc ML](#misc-ml)
- [Neuroscience](#neuroscience)

# Textbook
- [Sutton, Introduction to Reinforcement Learning, 2nd edition](/Sutton%20-%20Introduction%20to%20RL)

# Policy Gradients
- Silver et al., A2C
- Mnih et al., IMPALA
- Silver et al., DPG
- Silver et al., DDPG
- Kakade, Approximately Optimal Approximate Reinforcement Learning
- Kakade, A Natural Policy Gradient
- Schulman et al., TRPO
- Schulman et al., PPO
- Schulman, High-dimensional continuous control using generalized advantage estimation
- Wang et al., Sample Efficient Actor-Critic with Experience Reply
- Gu et al., Q-Prop
- Gruslys et al., The Reactor
- Liu et al., Stein Variational Policy Gradient
- Gu et al., Interpolated Policy Gradient: Merging On-Policy and Off-Policy Gradient Estimation for Deep Reinforcement Learning

# DQN
- Silver et al., Human-level control through deep reinforcement learning
- Silver et al. Double Q-Learning
- Silver et al. Dueling network architectures for deep reinforcement learning
- Silver et al. Prioritized experience replay
- Bellemare et al., A Distributional Perspective on Reinforcement Learning
- Silver et al., Rainbow: Combining Improvements in DRL

# Model-based RL & Planning
- Doll et al., The ubiquity of model-based reinforcement learning
- Tamar et al., [Value Iteration Networks](https://github.com/zuoxingdong/VIN_PyTorch_Visdom)
- Tamar et al., Learning Generalized Reactive Policies using Deep Neural Networks
- Tamar et al., Learning Plannable Representations with Causal InfoGAN
- Singh et al., Value Prediction Networks
- Lin et al., Value Propagation Networks
- Lee et al., Gated Path Planning Networks
- Salakhutdinov et al., LSTM Iteration Networks: An Exploration of Differentialble Path Finding
- Abbeel et al., Universal Planning Networks
- Wierstra et al., Learning Dynamic State Abstractions for Model-Based Reinforcement Learning
- Gal et al., Improving PILCO with Bayesian Neural Network Dynamics Models
- Meger et al., Synthesizing Neural Network Controllers with Probabilistic Model-based Reinforcement Learning
- Levine et al., Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models
- Wierstra et al., Learning model-based planning from scratch
- Gu et al., Continuous Deep Q-Learning with Model-based Acceleration
- Lecun et al., Model-Based Planning in Discrete and Continuous Actions
- Silver et al., The Predictron: End-To-End Learning and Planning
- Weber et al., Imagination-Augmented Agents for Deep Reinforcement Learning
- Li et al., Iterative Linear Quadratic Regulator Design for Nonlinear Biological Movement Systems
- Chockalingam et al., Differentiable Neural Planners with Temporally Extended Actions
- Mishra et al., Prediction and Control with Temporal Segment Models
- Metz et al., Discrete Sequential Prediction of Continuous Actions for Deep RL
- Moerland et al., Learning Multimodal Transition Dynamics for Model-Based Reinforcement Learning
- Chiappa et al., Recurrent Environment Simulators
- Vinyals et al., Metacontrol for adaptive imagination-based optimization
- Gerstner et al., Efficient Model-based Deep Reinforcement Learning with Variational State Tabulation
- Dinh et al., Learning Awareness Models
- Abbeel et al., Model-ensemble Trust-Region Policy Optimization
- Levine et al., Model-based Value Expansion for Efficient Model-Free Reinforcement Learning
- Levine et al., Recall Traces: Backtracking Models for Efficient Reinforcement Learning
- Levine et al., Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning
- Levine et al., Temporal Difference Models: Model-Free Deep RL for Model-Based Control
- Gregor et al., Temporal Difference Variational Auto-Encoder
- Abbeel et al., SOLAR: Deep Structured Latent Representations for Model-Based Reinforcement Learning
- Scholkopft et al., Adaptive Skip Intervals: Temporal Abstraction for Recurrent Dynamical Models
- Singh et al., Improving model-based RL with Adaptive Rollout using Uncertainty Estimation
- Abbeel et al., Model-Based Reinforcement Learning via Meta-Policy Optimization

# RL Theory
- Osband et al., A Tutorial on Thompson Sampling (Journal version, 2018)
- Osband et al. (More) efficient reinforcement learning via posterior sampling
- Osband et al., Why is Posterior Sampling Better than Optimism for Reinforcement Learning?
- Nachum et al., Bridging the Gap Between Valud and Policy Based Reinforcement Learning
- Bellemare et al., Increasing the Action Gap: New Operators for Reinforcement Learning
- Tishby et al., A Unified Bellman Equation for Causal Information and Value in Markov Decision Processes
- Dai et al., SBEED: Convergent Reinforcement Learning with Nonlinear Function Approximation
- Meger et al., Addressing Function Approximation Error in Actor-Critic Methods
- Schaul et al., Universal Value Function Approximators
- Levine, Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review

# Misc RL
- [Schmidhuber, PowerPlay: Training an increasingly general problem solver by continually searching for the simplest still unsolvable problem](/PowerPlay)
- [Dopamine: A Research Framework for Deep Reinforcement Learning](/dopamine)
- Salimans et al., Evolution Strategies as a Scalable Alternative to Reinforcement Learning
- Silver et al., Memory-based control with recurrent neural networks
- Rusu et al., Policy Distillation
- Schulman et al., Teacher-Student Curriculum Learning
- Rezende et al., Interaction Networks for Learning about Objects, Relations and Physics
- Silver et al., Learning Continuous Control Policies by Stochastic Value Gradients
- Silver et al., Continuous control with deep reinforcement learning
- Osband et al., Randomized Prior Functions for Deep Reinforcement Learning
- Clopath et al., Continual Reinforcement Learning with Complex Synapses
- Lin et al., Intrinsic Motivation and Automatic Curricula via Asymmetric Self-Play
- Oudeyer et al., Intrinsically Motivated Goal Exploration Processes with Automatic Curriculum Learning
- Oudeyer et al., Unsupervised Learning of Goal Spaces for Intrinsically Motivated Goal Exploration
- Sutton et al., Between MDPs and Semi-MDPs: Learning, planning, and representing knowledge at multiple temporal scales
- Silver et al., FeUdal Networks for Hierarchical Reinforcement Learning
- Silver et al., Meta-Gradient Reinforcement Learning
- Abbeel et al., Continuous Adaptation via Meta-Learning in Nonstationary and Competitive Environments
- Abbeel et al., Learning to Adapt: Meta-Learning for Model-Based Control
- Schulman et al., On First-Order Meta-Learning Algorithms
- Schaul et al., Learning to learn by gradient descent by gradient descent
- Abbeel et al., Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks
- Botvinick et al., Learning to Reinforcement Learn
- Osband et al. Deep Exploration via Bootstrapped DQN
- Abbeel et al., VIME: Variational Information Maximizing Exploration
- Ostrovski et al., Count-Based Exploration with Neural Density Models
- Tang et al., #Exploration: A Study of Count-based Exploration for Deep Reinforcement Learning
- Fortunato et al., Noisy Networks for Exploration
- Plappert et al., Parameter Space Noise for Exploration
- Bellemare et al., Unifying Count-Based Exploration and Intrinsic Motivation
- Levine et al., EX2: Exploration with Exemplar Models for Deep Reinforcement Learning
- Moerland et al., The Potential of the Return Distribution for Exploration in RL
- Pineau et al., Randomized Value Functions via Multiplicative Normalizing Flows
- Abbeel et al., Incentivizing Exploration In Reinforcement Learning With Deep Predictive Models
- Riedmiller et al., Learning by Playing-Solving Sparse Reward Tasks from Scratch
- Riedmiller et al., Maximum a Posteriori Policy Optimisation
- Riedmiller et al., Graph networks as learnable physics engines for inference and control

# Optimization & Variational Inference
- Bottou, Stochastic Gradient Descent Tricks
- Bottou et al., Optimization Methods for Large-Scale Machine Learning
- Martens et al., Optimizing Neural Networks with Kronecker-factored Approximate Curvature
- Barber et al., [Variational Optimization](/Variational_Optimization)
- Grathwohl et al., Backpropagation through the Void: Optimizing control variates for black-box gradient estimation
- Blei et al., Variational Inference: A Review for Statisticians
- Grosse et al., Noisy Natural Gradient as Variational Inference
- Whiteson et al., DiCE: The Infinitely Differentiable Monte-Carlo Estimator
- Maddison et al., The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables
- Gu et al., Categorical Reparameterization with Gumbel-Softmax
- Barber et al., Stochastic Variational Optimization
- Martens et al., New insights and perspectives on the natural gradient method

# Misc ML
- [Normalization tricks](/normalization_tricks/normalization_tricks.pdf)
  - Batch norm, layer norm
- Schon et al., Manipulating the Multivariate Gaussian Density
- Dumoulin, A guide to convolution arithmetic for deep learning
- Kingma et al., Auto-Encoding Variational Bayes
- Rusu et al., Progressive Neural Networks
- Kirkpatrick et al., Overcoming catastrophic forgetting in neural networks
- Bengio, Curriculum Learning
- Graves et al., Automated Curriculum Learning for Neural Networks
- Blundell et al., Weight Uncertainty in Neural Networks
- Blundell et al., Bayesian Recurrent Neural Networks
- Gal et al., What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision ?
- Hernandez-Lobato et al., Black-Box alpha-Divergence Minimization
- Roeder et al., Sticking the Landing: Simple, Lower-Variance Gradient Estimators for Variational Inference
- Lakshminarayanan et al., Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles
- Vetrov et al., Structured Bayesian Pruning via Log-Normal Multiplicative Noise
- Rezende et al., Neural Processes

# Neuroscience
- Hassabis et al., Neuroscience-Inspired Artificial Intelligence
- Tenenbaum et al., Building machines that learn and think like people
- Doll, The ubiquity of model-based reinforcement learning
- Moser et al., Place cells, grid cells, and the brain's spatial representation system
- Niv et al., Reinforcement learning in the brain
