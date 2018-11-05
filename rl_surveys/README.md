![](./.png)

- Sample efficiency
  - Prior knownledge: demonstration
  - Extract more information from observations:  learning a model
  - Sample reuse: experience 

- Policy search without utility models: sample policy parameters and move towards direction for higher utility
  - Random search: randomly searches parameters until good enough
  - Inefficient
    - No assumption on objective function
    - Without estimating any gradients
  - Population-based optimization: maintain a population and sample individuals based on previously elite set. 
  - Evolution strategies (ES): specific population-based optimization
    - Compute optimum guess from samples in previous generation
    - Samples in new generation are obtained by adding Gaussian noise to current optimum guess. 
    - Estimation of Distribution Algorithms (EDAs): specific ES by maintaining a covariance matrix e.g. CEM, CMA-ES

- Model-based policy search: learning a transition dynamics model over the observed trajectories
  - Deterministic model: suffer from model bias e.g. neural networks
  - Probabilistic model: distribution of all plausible models e.g. Gaussian processes, Bayesian neural networks
  - Bayesian optimization (BO): actively choose next observation by using acquisition functions e.g. UCB, PI, EI

- Directed explorations: 
