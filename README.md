# 🌊 Bayesian Search & Rescue: A Python Simulation

This repository contains a Python project that demonstrates Bayesian inference applied to a search and rescue (SAR) problem. The project evolves in three stages, from a simple probability simulation to an advanced strategic game with a dynamic, moving target.

The basic project originates from the book “Real World Python” from Lee Vaughan.

It uses **OpenCV** for visualization and **NumPy** for probability calculations.



---

## 🚀 Project Evolution

This repository contains three distinct versions of the code, each building upon the last.

### 1. Basic Simulation (`bayes_codes_basic.py`)
A simple, static demonstration of the core Bayesian updating concept.

* **Static Map:** Loads a map image with three pre-defined search areas (SAs).
* **Static Probabilities:** The game always starts with fixed initial probabilities (e.g., P(Area 1) = 0.2, P(Area 2) = 0.5, P(Area 3) = 0.3).
* **Static Target:** The sailor's location is randomly generated *once* at the beginning and **does not move**.
* **Core Bayesian Logic:** When a search in an area fails, the probability for that area decreases, and the probabilities for all other areas are revised upwards.

### 2. Intermediate: Strategic Game (`bayes_codes_intermediate.py`)
This version turns the simulation into a true game with tension, strategy, and replayability.

* **All Basic Features +**
* **Turn Limit:** Introduces tension. The player has a limited number of turns (e.g., 10 "days" of fuel) to find the sailor. **It is now possible to lose the game.**
* **Area Difficulty:** Each area now has a different "Search Effectiveness" (SEP) range.
    * **Area 1 (Hard):** Low SEP (e.g., rocky, foggy).
    * **Area 2 (Easy):** High SEP (e.g., open ocean).
    * **Area 3 (Medium):** Moderate SEP.
* **Randomized Start:** The initial probabilities are randomized at the start of every game, forcing the player to adapt their strategy and ensuring no two games are identical.

### 3. Advanced: Dynamic Simulation (`bayes_codes_advance.py`)
This version models a more realistic scenario by introducing a target that is no longer static.

* **All Intermediate Features +**
* **Moving Target:** The sailor is now "adrift." The target's **physical location** has a chance to move from one area to another each turn, simulating ocean currents.
* **Markov Chain Drift:** The *probabilities* themselves are updated each turn using a **Markov transition matrix**. This models the "flow" of probability *before* the player even makes a move.
* **Automatic Game Loop:** After a win or loss, the game automatically resets and starts a new, randomized round.

---

## 🧠 Core Concepts

This project is a practical demonstration of two key probabilistic models.

#### 1. Bayesian Inference (All Versions)
This is used to **update our beliefs based on new evidence**. We answer the question:

> "What is the new probability that the sailor is in Area 1, **given that** our search in Area 1 just **failed**?"

We use the Search Effectiveness (SEP) as the likelihood $P(\text{Fail}|A)$ and update our prior probabilities accordingly. The formula for a failed search is:


$$P(A|\text{Fail}) = \frac{P(\text{Fail}|A) \cdot P(A)}{P(\text{Fail})}$$


#### 2. Markov Chains (Advanced Version)
This is used to **model the passage of time**. It describes the probability of moving from one state (e.g., Area 1) to another (e.g., Area 2) in a single step (one turn).

We use a transition matrix to calculate how the probabilities change *before* we search:

$$\begin{bmatrix} P(1)_{\text{new}} & P(2)_{\text{new}} & P(3)_{\text{new}} \end{bmatrix} = \begin{bmatrix} P(1)_{\text{old}} & P(2)_{\text{old}} & P(3)_{\text{old}} \end{bmatrix} \times \begin{bmatrix} P(1\to 1) & P(1\to 2) & P(1\to 3) \\ P(2\to 1) & P(2\to 2) & P(2\to 3) \\ P(3\to 1) & P(3\to 2) & P(3\to 3) \end{bmatrix}$$


The advanced game loop combines these two concepts into a powerful cycle:
**1. Apply Drift (Markov)** $\rightarrow$ **2. Player Searches** $\rightarrow$ **3. Update Beliefs (Bayes)**

---

## 📋 Requirements

* Python 3.x
* NumPy (`pip install numpy`)
* OpenCV for Python (`pip install opencv-python`)

---

## ⚙️ How to Use

1.  Clone the repository:
    ```bash
    git clone [https://github.com/moisesdatasci/bayes_code]
    cd your-repository-name
    ```

2.  Install the dependencies:
    ```bash
    pip install numpy opencv-python
    ```

3.  Make sure the map file `cape_python.png` is in the same directory as the scripts.

4.  Run your desired version:
    ```bash
    # To run the simple simulation
    python basic.py

    # To play the strategic game
    python intermediate.py

    # To play the advanced simulation with a moving target
    python advanced.py
    ```
