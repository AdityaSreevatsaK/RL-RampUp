# <p align="center">Reinforcement Learning - Basics</p>
## 1. **What is Reinforcement Learning?**
Reinforcement Learning is a type of machine learning where an `agent` learns to make decisions by interacting with an 
`environment`. The agent learns what actions yield the most `reward` by **trail and error**. <br />

---

**Key Contrast:**
- Supervised Learning: Learns from labelled data.
- Unsupervised Learning: Finds patterns in data.
- Reinforcement Learning: Learns by acting, observing outcomes (rewards), and adjusting behaviour.

---

## 2. **Key Concepts**
### Agent

* **Definition:** The agent is the entity that interacts with the environment. It observes states, selects actions, and 
learns to improve its behavior over time.
* **Examples:**

  * A self-driving car (agent) driving on roads (environment).
  * An AI playing chess (agent) on a chessboard (environment).
* **Goal:** Maximize total cumulative reward collected over time.

---

### Environment

* **Definition:** The outside world that the agent interacts with. It defines the rules, dynamics, and feedback for the
agent.
* **Responsibilities:**

  * Provides observations (states) to the agent.
  * Responds to agent’s actions with new states and rewards.
  * Determines when an episode ends (terminal state).
* **Examples:**

  * The grid and rules in a maze game.
  * The stock market with price movements.
  * The board and rules of a video game.

---

### State (s)

* **Definition:** A representation of the current situation of the environment, as perceived by the agent.
* **Can be:**

  * Fully observable (the agent sees the complete state, e.g., board in chess).
  * Partially observable (agent only sees part, e.g., only nearby cars in driving).
* **Form:**

  * Discrete (like squares in a grid).
  * Continuous (like position and velocity of a car).
* **Purpose:** The state provides all the information the agent uses to decide its next action.

---

### Action (a)

* **Definition:** A decision made by the agent that influences the environment.
* **Types:**

  * **Discrete:** A finite set (left, right, jump, hold, etc.).
  * **Continuous:** Any value within a range (steering angle, amount to invest, etc.).
* **Action Space:** The set of all possible actions.
* **Role:** Each action taken by the agent affects the next state and the reward received.

---

### Reward (r)

* **Definition:** A single number (scalar) given by the environment to the agent after each action.
* **Purpose:** Indicates the immediate value of the last action. It's the only feedback the agent directly gets about 
its performance.
* **Design:**

  * **Sparse:** Only given at the end or at rare events (e.g., win/loss).
  * **Dense:** Frequent feedback (e.g., every step).
* **Goal:** The agent tries to maximize the sum of rewards over time (cumulative reward).

---

### Policy (π)

* **Definition:** The agent’s behavior function: a mapping from states to actions (or to probabilities of actions).
* **Formally:**

  * Deterministic: $a = \pi(s)$
  * Stochastic: $a \sim \pi(\cdot|s)$ (probability distribution over actions given state)
* **Learned or Fixed:** Can be hard-coded, learned through trial-and-error, or a mix.
* **Goal:** Learn an optimal policy that yields the highest expected reward over time.

---

### Episode

* **Definition:** A sequence of states, actions, and rewards, from the starting state to a terminal (ending) state.
* **Examples:**

  * One full game of chess (from start to checkmate).
  * One run in CartPole (until the pole falls or time limit is reached).
* **Importance:** Many RL algorithms learn from complete episodes to evaluate cumulative returns.

---

### Value Function (V)

* **Definition:** Estimates the **expected cumulative reward** (sum of future rewards), starting from a state
(or state-action pair) and following a policy.
* **Mathematically:**

  * $V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{t=0}^\infty \gamma^t r_{t+1} \mid s_0 = s \right]$

    * $\gamma$: discount factor for future rewards (0 < γ < 1).
* **Purpose:** Helps the agent evaluate how “good” a state is.

---

### Q-Value or Action-Value Function (Q)

* **Definition:** Estimates the expected cumulative reward for **taking action a in state s**, and then following 
policy π.
* **Mathematically:**

`Q^π(s, a) = E_π [ sum_{t=0}^∞ γ^t r_{t+1} | s₀ = s, a₀ = a ]`


* **Usage:**

  * Central to algorithms like Q-learning and Deep Q Networks (DQN).
  * The agent chooses actions with the highest Q-value for a given state.

---

### Discount Factor (γ) *(Bonus Concept)*

* **Definition:** A number between 0 and 1 used to discount future rewards. Immediate rewards are valued more than 
distant rewards.
* **Why:** Encourages the agent to prefer quick wins.
* **Typical values:** 0.9, 0.99, etc.

---

### Summary Table

| Concept       | Description                                                             | Example                           |
|---------------|-------------------------------------------------------------------------|-----------------------------------|
| Agent         | The learner/decision maker                                              | Robot, game AI                    |
| Environment   | The world agent interacts with                                          | Game board, simulation            |
| State         | The current situation of the environment                                | Chessboard config                 |
| Action        | What the agent can do                                                   | Move left, right, up, down        |
| Reward        | Feedback on agent’s action                                              | +1 for win, -1 for loss           |
| Policy        | Mapping from states to actions                                          | “If pole falls left, push left”   |
| Episode       | One complete sequence from start to finish                              | One full chess game               |
| Value Fn.     | How good it is to be in a given state (expected future reward)          | “Is this chess position good?”    |
| Q-Value Fn.   | How good is a state-action pair (expected reward for action in state)   | “Is this move good now?”          |
---

## 3. **How RL Works (the RL loop)**

1. The agent observes the current **state** of the environment.
2. The agent chooses an **action** based on its **policy**.
3. The action changes the **environment**.
4. The agent receives a **reward** and observes the new **state**.
5. Steps 1-4 repeat, and the agent updates its policy to maximize cumulative reward.

---

## 4. **Mathematical Foundations**

RL often uses the **Markov Decision Process (MDP)**:

* **States (S)**
* **Actions (A)**
* **Transition function (P)**: Probability of moving to a new state.
* **Reward function (R)**
* **Discount factor (γ)**: Future rewards are worth less than immediate rewards.

---

## 5. **Types of RL Algorithms**

### **1. Value-Based Methods**

* Learn a value function (e.g., Q-Learning, Deep Q-Networks/DQN)

### **2. Policy-Based Methods**

* Directly learn the policy (e.g., REINFORCE, Policy Gradient Methods)

### **3. Actor-Critic Methods**

* Combine value-based and policy-based (e.g., A2C, PPO)

### **4. Model-Based RL**

* Learn a model of the environment to plan ahead.

---

## 6. **Key RL Algorithms**

1. **Q-Learning:**

   * Learns a Q-table for state-action values.
   * Good for small, discrete environments.

2. **Deep Q-Networks (DQN):**

   * Uses neural nets to approximate Q-values for large or continuous state spaces.

3. **Policy Gradient Methods:**

   * Directly optimize the policy via gradient ascent.

4. **Actor-Critic Methods:**

   * Maintain both a policy (actor) and value function (critic).

---

## 7. **Challenges in RL**

* **Exploration vs. Exploitation:** Should the agent try new actions (explore) or use known ones (exploit)?
* **Sample Efficiency:** RL usually needs a lot of data.
* **Stability & Convergence:** Harder than in supervised learning.
* **Reward Design:** The agent is only as clever as your reward function.

---

## 8. **Classic RL Environments (for practice)**

* **OpenAI Gym:** Collection of environments like CartPole, MountainCar, FrozenLake.
* **Atari Games:** For deep RL research.
* **Custom Simulations:** Gridworld, maze, robotics.

---

## 9. **Suggested Learning Path**

1. **Start with the basics:**

   * Multi-armed bandit problem
   * Tabular Q-learning (simple gridworld, tic-tac-toe)
2. **Move to OpenAI Gym environments**

   * CartPole, MountainCar
3. **Implement Deep Q-Networks**
4. **Explore Policy Gradients and Actor-Critic methods**
5. **Experiment with custom environments and reward functions**

---

## 10. **Recommended Resources**

* **Books:**

  * *Reinforcement Learning: An Introduction* by Sutton & Barto
  * *Deep Reinforcement Learning Hands-On* by Maxim Lapan

* **Courses:**

  * David Silver’s RL course (free, YouTube)
  * OpenAI Spinning Up (great practical resource)

* **Frameworks:**

  * `gymnasium` (successor of OpenAI Gym)
  * `stable-baselines3` (for RL algorithms)

---
