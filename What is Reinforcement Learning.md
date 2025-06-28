# <p align="center">Reinforcement Learning - Basics</p>
## 1. **What is Reinforcement Learning?**
Reinforcement Learning is a type of machine learning where an `agent` learns to make decisions by interacting with an `environment`. The agent learns what actions yield the most `reward` by **trail and error**. <br />

---

**Key Contrast:**
- Supervised Learning: Learns from labelled data.
- Unsupervised Learning: Finds patterns in data.
- Reinforcement Learning: Learns by acting, observing outcomes (rewards), and adjusting behaviour.

---

## 2. **Key Concepts**
### Agent

* **Definition:** The agent is the entity that interacts with the environment. It observes states, selects actions, and learns to improve its behavior over time.
* **Examples:**

  * A self-driving car (agent) driving on roads (environment).
  * An AI playing chess (agent) on a chessboard (environment).
* **Goal:** Maximize total cumulative reward collected over time.

---

### Environment

* **Definition:** The outside world that the agent interacts with. It defines the rules, dynamics, and feedback for the agent.
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

  * Deterministic: `a = \pi(s)`
  * Stochastic: `a \sim \pi(\cdot|s)` (probability distribution over actions given state)
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

  `V^π(s) = E_π [ sum_{t=0}^∞ γ^t r_{t+1} | s₀ = s ]`

    * `γ` (Gamma): discount factor for future rewards (0 < γ < 1).
* **Purpose:** Helps the agent evaluate how “good” a state is.

---

### Q-Value or Action-Value Function (Q)

* **Definition:** Estimates the expected cumulative reward for **taking an action `a` in a state `s`**, and then 
following policy `π`.
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

### Value-Based Methods

**Goal:**
Learn a function that estimates how good it is to be in a state, or to take a certain action in a state (i.e., the value function or Q-function).

**How they work:**

* The agent builds a table or function mapping state-action pairs to expected cumulative rewards (“Q-values”).
* The agent follows the **greedy policy**: always picks the action with the highest value (with some exploration).

**Classic Examples:**

* **Q-Learning:**

  * Learns the optimal action-value function `Q^*(s, a)` using a simple update rule.
  * Works well for small, discrete environments (where all states/actions can fit in a table).
* **SARSA:**

  * Similar to Q-Learning, but is “on-policy” (updates Q-values based on the action the current policy would actually take).
* **Deep Q-Networks (DQN):**

  * Extends Q-Learning to complex, high-dimensional environments by using neural networks to approximate Q-values.
  * Famous for mastering Atari games from raw pixels.

**When to use:**

* When the state/action spaces are not too huge, or can be approximated.
* When you care about learning which actions are good in which states, not just what to do.

---

### Policy-Based Methods

**Goal:**
Learn the policy **directly**—a mapping from states to actions—rather than first learning value functions.

**How they work:**

* The agent’s policy is typically a neural network (for continuous spaces).
* The policy is optimized directly (using gradient ascent) to maximize expected rewards.

**Classic Examples:**

* **REINFORCE:**

  * A basic policy gradient method.
  * Collects episodes, computes returns, and updates policy weights in the direction that increases probability of actions with higher returns.
* **Policy Gradient Methods:**

  * Broader class including advanced algorithms (e.g., TRPO, PPO).

**Strengths:**

* Can handle continuous and/or high-dimensional action spaces.
* Can learn stochastic policies (where actions are sampled from a probability distribution).

**Weaknesses:**

* Usually have higher variance in updates (less stable).
* May be less sample efficient than value-based methods.

**When to use:**

* When the action space is continuous (e.g., robotics, finance).
* When you need a stochastic policy.

---

### Actor-Critic Methods

**Goal:**
Combine the strengths of value-based and policy-based methods.

**How they work:**

* There are two networks (“heads”):

  * **Actor:** Proposes actions (the policy).
  * **Critic:** Estimates the value function (criticizes the actor’s choices).
* The critic helps reduce the variance of the policy gradient, making learning more stable.

**Classic Examples:**

* **Advantage Actor-Critic (A2C, A3C):**

  * Synchronous/asynchronous versions of the basic actor-critic approach.
* **Proximal Policy Optimization (PPO):**

  * Modern, stable, and popular; uses clever tricks to ensure updates are not too large.
* **Deep Deterministic Policy Gradient (DDPG), Soft Actor-Critic (SAC):**

  * Handle continuous action spaces.

**Strengths:**

* More stable and sample-efficient than plain policy gradient.
* Scales well to large problems.

**When to use:**

* When you want stability and scalability.
* When value-based or policy-based methods alone are too unstable or inefficient.

---

### Model-Based RL

**Goal:**
Learn a model of the environment’s dynamics (i.e., how the environment transitions from state to state given actions) and use it to plan or simulate possible futures.

**How they work:**

* The agent learns:

  * A **transition model:** `P(s'|s, a)`
  * A **reward model:** `R(s, a)`
* The agent can “imagine” or simulate future trajectories using the model, improving sample efficiency.
* Planning algorithms (like **Model Predictive Control** or **Monte Carlo Tree Search**) can be used on the model.

**Classic Examples:**

* **Dyna-Q:**

  * Blends model-free Q-learning with a simple learned model for planning.
* **AlphaZero:**

  * Combines deep learning with Monte Carlo Tree Search to play Go, Chess, and Shogi at superhuman levels.

**Strengths:**

* Can be much more sample efficient, since the agent can train on imagined experiences.
* Enables planning ahead, not just reactive behaviors.

**Weaknesses:**

* Harder to implement and tune; learning an accurate environment model can be very challenging, especially in complex domains.
* Errors in the model can lead to bad decisions (“model bias”).

**When to use:**

* When environment interactions are expensive or slow (e.g., real-world robotics).
* When you can’t afford to learn only by trial and error.

---

### Summary Table
| Method       | What is learned?       | Pros                                    | Cons                           | Example Algorithms     |
|--------------|------------------------|-----------------------------------------|--------------------------------|------------------------|
| Value-Based  | Value functions/Q(s,a) | Simple, stable                          | Can’t handle continuous action | Q-Learning, DQN        |
| Policy-Based | Policy directly        | Works for continuous/stochastic actions | High variance, less stable     | REINFORCE, Policy Grad |
| Actor-Critic | Both policy and value  | Stable, efficient                       | More complex                   | A2C, PPO, DDPG, SAC    |
| Model-Based  | Model of environment   | Sample efficient, can plan              | Model errors can be fatal      | Dyna-Q, AlphaZero      |

---

## 6. **Key RL Algorithms**

### Q-Learning

**What it is:**
A foundational, model-free, value-based RL algorithm for learning the optimal action-selection policy for an agent.

**How it works:**

* Maintains a **Q-table**: a table where each row is a state, each column is an action, and each cell stores the current estimate of the “quality” (expected cumulative reward) of taking that action in that state.
* At each step:

  1. The agent observes the current state.
  2. Chooses an action (using an exploration strategy, e.g., ε-greedy).
  3. Receives a reward and observes the next state.
  4. Updates the Q-value for the (state, action) pair using the **Q-learning update rule**:

     ```
     Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]
     ```

     Where:

     * `α` is the learning rate,
     * `γ` is the discount factor,
     * `r` is the reward,
     * `s'` is the next state,
     * `max_a' Q(s', a')` is the maximum estimated value for the next state.

**Strengths:**

* Simple and easy to implement.
* Proven to converge to the optimal policy if every action is sampled enough in every state (with decaying exploration).

**Limitations:**

* Only works well in environments with **discrete** (finite) state and action spaces.
* The Q-table grows quickly with the size of the state/action spaces (curse of dimensionality).

**Typical use:**
Simple games (FrozenLake, GridWorld, Taxi-v3), toy problems.

---

### Deep Q-Networks (DQN)

**What it is:**
An extension of Q-learning that uses a **deep neural network** to approximate the Q-function, making it scalable to environments with large or continuous state spaces (like images).

**How it works:**

* Instead of a Q-table, a neural network (`Q(s, a; θ)`) predicts Q-values for all actions given a state.
* The agent collects experience in the form of (state, action, reward, next\_state) tuples.
* These experiences are stored in a **replay buffer**; during training, random samples (“mini batches”) are used to break correlations and stabilize training.
* Uses a **target network** to further stabilize learning (the target network is updated less frequently than the main 
network).

**Key innovations:**

* **Experience Replay:** Learn from a random sample of experiences, not just most recent.
* **Target Network:** Reduce instability by using a slowly-updated separate network to generate target Q-values.

**Strengths:**

* Handles very large or high-dimensional state spaces (e.g., pixels in Atari games).
* The basis for many advanced RL approaches.

**Limitations:**

* Only works for **discrete** action spaces (out of the box).
* Can be hard to tune (hyperparameters, neural net architecture).
* Sample inefficient compared to some other approaches.

**Typical use:**
Playing Atari games, board games with large state spaces.

---

### Policy Gradient Methods

**What they are:**
A family of RL algorithms that **directly optimize the policy** (how the agent acts), rather than first learning values for states or actions.

**How they work:**

* The agent’s policy is represented as a parameterized function (e.g., a neural network with weights θ), which outputs the probability of each action given a state.
* The objective is to **maximize the expected reward** directly by adjusting the parameters of the policy in the 
direction that increases the probability of actions that led to higher returns.
* Updates are performed using stochastic gradient ascent on the expected reward.

**A basic algorithm: REINFORCE**

* Collect a batch of trajectories (state-action-reward sequences).
* For each action taken, compute how much total reward was obtained.
* Increase the probability of actions that led to high returns, decrease for low returns.

**Strengths:**

* Work with **continuous** or **high-dimensional** action spaces.
* Can learn **stochastic** policies (important when randomness is needed).
* More flexible: can incorporate constraints or multi-objective optimization.

**Limitations:**

* High variance in updates (can be unstable, slow learning).
* Require careful tuning and large batches for stable learning.
* Sometimes less sample-efficient than value-based methods.

**Typical use:**
Robotics, games with continuous action spaces (e.g., MuJoCo, robotics simulation tasks).

---

### Actor-Critic Methods

**What they are:**
Hybrid methods that combine ideas from value-based and policy-based approaches for improved efficiency and stability.

**How they work:**

* Maintain **two separate models**:

  * **Actor**: The policy (decides what to do; outputs action probabilities).
  * **Critic**: The value function (estimates how good a given state or state-action pair is; helps the actor update more efficiently).
* The critic estimates how much better (or worse) an action is compared to average (the “advantage”). The actor then uses this feedback to update the policy.

**Popular variants:**

* **Advantage Actor-Critic (A2C, A3C):** Estimate the “advantage” function to reduce variance in policy gradient updates.
* **Proximal Policy Optimization (PPO):** Uses clipped objective functions for even more stable updates; widely used and state-of-the-art for many tasks.
* **Deep Deterministic Policy Gradient (DDPG):** Adapts actor-critic for continuous actions.
* **Soft Actor-Critic (SAC):** Adds entropy maximization for improved exploration in continuous action spaces.

**Strengths:**

* Combine stability and efficiency: lower variance than pure policy gradient, more flexible than value-only methods.
* Work well in large, continuous environments.
* Modern RL research uses these methods extensively.

**Limitations:**

* More complex to implement and tune (need to balance two models).
* Sensitive to hyperparameters and learning rate schedules.

**Typical use:**
Challenging simulated tasks (robotics, OpenAI Gym, video games), continuous control problems.

---

### Quick Comparison Table

| Algorithm       | Key Idea                                    | Pros                         | Cons                        | Example Use         |
|-----------------|---------------------------------------------|------------------------------|-----------------------------|---------------------|
| Q-Learning      | Learn Q-table for discrete states/actions   | Simple, proven               | Doesn’t scale to big spaces | Gridworld, Taxi-v3  |
| Deep Q-Networks | Neural net Q-function for big state spaces  | Handles images, large states | Discrete actions only       | Atari games         |
| Policy Gradient | Directly optimize policy (actions)          | Handles continuous actions   | High variance, slow         | Robotics (MuJoCo)   |
| Actor-Critic    | Both policy (actor) & value (critic) models | Stable, efficient, scalable  | More complex to code/tune   | PPO, SAC, A2C, DDPG |

---

## 7. **Challenges in RL**

### Exploration vs. Exploitation

**What’s the issue?**

* The agent must balance **exploring** new actions to find potentially better rewards, and **exploiting** the best-known action to maximize immediate rewards.
* If it explores too little, it might get stuck in a suboptimal behavior (“local optimum”). If it explores too much, it wastes time and resources not taking the best-known actions.

**Classic Example:**

* Multi-armed bandit problem: Which slot machine do you play to maximize your winnings? You have to try each one (explore), but also want to play the best one once you find it (exploit).

**In RL algorithms:**

* Commonly handled with **ε-greedy** policies: with probability ε, take a random action (explore); with probability 1-ε, take the best action (exploit).
* Other approaches: **Boltzmann exploration**, **Upper Confidence Bound (UCB)**, **entropy regularization** (especially in policy gradient methods), or **curiosity-driven exploration** (agent gets extra reward for visiting novel states).

**Why it’s hard:**

* In environments with **sparse rewards**, exploration becomes much more important, as the agent can go a long time without any feedback.
* There’s often no “right” ε value—the ideal balance can change over time.

---

### Sample Efficiency

**What’s the issue?**

* RL agents often need **huge amounts of experience** (millions of steps) to learn effective policies, especially in complex environments.
* This is very different from supervised learning, where you often learn from a fixed dataset.

**Why is RL sample inefficient?**

* Data is generated on-the-fly as the agent interacts with the environment.
* Each experience depends on the current policy, so the data distribution is non-stationary.
* Many RL algorithms, especially value-based methods, update the agent’s knowledge slowly over many experiences.

**Real-world consequences:**

* Training agents in physical robots, real financial systems, or any environment where data collection is costly can be impractical.

**Strategies to improve sample efficiency:**

* **Experience Replay:** Store past experiences and learn from them repeatedly (used in DQN).
* **Model-Based RL:** Learn a model of the environment to generate “imaginary” experience.
* **Transfer Learning:** Reuse knowledge from similar tasks.
* **Imitation Learning:** Pre-train the agent on expert demonstrations before RL fine-tuning.

---

### Stability & Convergence

**What’s the issue?**

* RL algorithms can be **unstable** and may not always converge to a good solution, especially compared to supervised learning.
* Small changes in hyperparameters, network architecture, or even random seeds can lead to very different outcomes.

**Sources of instability:**

* **Bootstrapping:** Many algorithms update estimates based on other estimates (e.g., Q-learning uses max Q-value from the next state to update the current Q-value), leading to feedback loops.
* **Function approximation:** Using neural networks to approximate value functions (as in DQN) introduces bias and variance.
* **Non-stationary data:** The distribution of experiences changes as the policy improves.
* **Delayed rewards:** Feedback for a decision may arrive many steps later, making credit assignment difficult.

**What’s done about it?**

* **Target Networks:** DQN uses a separate, slowly-updated target network to stabilize updates.
* **Batch Normalization and Reward Normalization:** Smooth out learning signals.
* **Careful Hyperparameter Tuning:** Learning rate, batch size, and architecture choices matter a lot.
* **Algorithmic advances:** PPO and SAC are designed to improve stability.

---

### Reward Design

**What’s the issue?**

* The reward function defines the agent’s objective. If it’s poorly designed, the agent may learn undesired behaviors or exploit loopholes in your task definition (“reward hacking”).

**Examples:**

* In a racing game, if you reward the agent only for speed, it might drive in circles near the start line to maximize speed, instead of finishing the lap.
* In a robotic walking task, if you reward only for forward movement, the agent might learn to fall forward instead of walking.

**Challenges:**

* **Sparse rewards:** If rewards are given only for achieving the end goal, learning can be very slow.
* **Shaping rewards:** Giving intermediate rewards (for progress toward the goal) can speed up learning, but may bias the agent toward suboptimal strategies.

**Best practices:**

* **Iterate:** Expect to refine your reward function several times.
* **Monitor agent behavior:** Watch for unintended solutions.
* **Use auxiliary rewards:** Sometimes adding curiosity or “intrinsic motivation” (rewarding the agent for visiting novel states) can help.
* **Inverse RL:** Learn reward functions from expert demonstrations instead of hand-crafting them.

---

### Summary Table

| Challenge                    | What it is                                        | Why it’s hard                         | How to address                          |
|------------------------------|---------------------------------------------------|---------------------------------------|-----------------------------------------|
| Exploration vs. Exploitation | Balancing trying new actions and using best-known | Too much/too little exploration hurts | ε-greedy, UCB, entropy, curiosity       |
| Sample Efficiency            | How much experience is needed                     | RL usually needs a ton of data        | Experience replay, model-based RL       |
| Stability & Convergence      | Whether training is robust and converges          | RL can be very unstable and brittle   | Target nets, normalization, PPO/SAC     |
| Reward Design                | Designing the feedback the agent gets             | Bad rewards = weird behaviors         | Iterate, monitor, use auxiliary rewards |

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
