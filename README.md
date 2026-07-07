# Deep Learning Applications

Repository containing the laboratories and projects developed for the **Deep Learning Applications** course.  
The repository covers topics ranging from Computer Vision to Reinforcement Learning, with experiment tracking and interactive reports available through **Weights & Biases**.

---

# 🎥 Project Presentation Videos

Short presentations.

- Video 1 → Coming soon
- Video 2 → Coming soon

---

# 📚 Repository Structure

```text
lab1/    -> Convolutional Neural Networks
lab3/    -> Reinforcement Learning
lab4/    -> OOD
```

---

# 🧠 Lab 1 - Convolutional Neural Networks

Brief description of the laboratory goals and implemented models.

## Weights & Biases Report

[Open W&B Report](#)

## Topics

- ...
- ...
- ...

## Experiments

- ...
- ...
- ...

## Results

Summary of the main findings and comparisons.

---

# 🤖 Lab 3 - Reinforcement Learning

This laboratory focuses on Deep Reinforcement Learning methods for solving control problems with discrete actions.  
We work on CartPole-v1 and LunarLander-v3 (both have discrete action spaces).  
The project starts from a basic implementation of REINFORCE and progressively introduces baselines and actor-critic methods.



<details>
<summary>How to run the project</summary>

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the main experiment:

```bash
python lab3/main.py
```

</details>

<details>
<summary>Project structure</summary>

```text
lab3/
│
├── src/
│   ├── networks.py
│   │   └── Policy network architectures
│   ├── reinforce.py
│   │   └── REINFORCE rollout and training loop
│   ├── utils.py
│   │   └── Helper functions and seed control
│   ├── a2c.py
│   │   └── Advantage Actor-Critic implementation
│   └── logger.py
│       └── Weights & Biases logging utilities
│
├── checkpoints/
│   └── Saved model checkpoints
├── config.py
│   └── Experiment hyperparameters
├── main.py
├── requirements.txt
│   └── Project dependencies
└── README.md
    └── Project documentation
```

</details>

## Experiments

The full experiment tracking, interactive plots, and comments are available in the W&B report:

[Open W&B Report](https://api.wandb.ai/links/pasqem-university-of-florence/7yw1qhbf)


## Results

Metric definitions used in the plots (Lab 3):

- **Training reward (return)**: sum of rewards over one training episode
    $$R_{\text{ep}} = \sum_{t=0}^{T-1} r_t$$

- **Training length**: number of steps per training episode
    $$T = \text{steps in the episode}$$

- **Eval average reward**: mean reward over $M$ evaluation episodes, computed every $N$ training episodes
    $$\text{eval\_avg\_reward} = \frac{1}{M}\sum_{i=1}^{M} R_i$$

- **Eval average length**: mean episode length over $M$ evaluation episodes, computed every $N$ training episodes
    $$\text{eval\_avg\_length} = \frac{1}{M}\sum_{i=1}^{M} T_i$$

- **Training losses**: diagnostic metrics used to monitor optimization.  
  For REINFORCE, we track the policy loss and the value loss of the learned baseline. For A2C, we track actor and critic losses separately: the actor loss describes policy updates based on the estimated advantages, while the critic loss measures the value-function approximation error.  
  Since reinforcement learning losses are often noisy and not directly comparable across algorithms, performance is mainly evaluated using reward and episode length metrics.

---

# ⚙️ Lab 4 - OOD

Brief description of the laboratory goals and implemented methods.

## Weights & Biases Report

[Open W&B Report](#)

## Topics

- ...
- ...
- ...

## Experiments

- ...
- ...
- ...

## Results

Summary of the main findings and comparisons.


---
# 🚀 Running Experiments

Example commands for running the projects.

```bash
# Lab 1
python ...

# Lab 2
python ...

# Lab 3
python ...
```

---

# 📌 Notes

Additional comments, observations, or future improvements.