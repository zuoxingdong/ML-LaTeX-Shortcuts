# Nuts and Bolts in Deep Reinforcement Learning

## Experimentation:
- Start with simple, toy task
- Fast prototype, hyperparameter search
- Unit test on common building blocks: e.g. discounted returns, TD, GAE etc. 
- Try multiple tasks, don't get stuck with specific one
- Visualizing many metrics and interpret them: e.g. gradient norm, policy entropy, value loss, explained variance etc.
- Diagnotics: episodic return (mean/std/min/max), episodic length
- No-free lunch: construct toy task emphasizing your idea
- Normalization over observation/rewards: VERY IMPORTANT, using running average or Chan's online algorithm
  - Clip scaled observation: x' = clip(scaled_x, -10, 10)
  - Rescale reward: similar, but don't shift the mean, otherwise it changes the problem
  - Standardize state-value targets
  - Why needed: e.g. some observation dimension has range (-100, 100) and some others have (-0.1, 0.1), then it's hard for NN to handle it
- Random seeds:
  - On-policy: 3 random seeds, use training data as performance evaluation
  - Off-policy: 10 random seeds, independent evaluation
- Architecture: LayerNorm + ReLU maybe better than tanh layer
- Entropy drops prematurely: no learning -> add entropy bonus
- Initialization: final layer with zero or small -> max entropy initially, e.g. Categorical/Gaussian distribution
- Automate experiment: don't waste time on watching numbers from screen

## Reproduce other baselines:
- Default setting mentioned in the paper might fail to work, maybe some *secret* tricks, or got some tiny thing wrongly
- Train more exhaustively, tune hyperparameters

## Misc advice:
- Read textbooks, not just recent papers: books contain more intensive information, recent paper only has one idea

# References:
- https://www.youtube.com/watch?v=8EcdaCk9KaQ
