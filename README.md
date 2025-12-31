# 🌀 Collatz-Lyapunov: Discovering the Hidden Gradient in Chaos

> **"Can one of the most famously 'unsolvable' problems in mathematics be used to build a better machine learning optimizer?"**

This project bridges the gap between **Discrete Dynamical Systems** and **Continuous Optimization**. By training a Neural Network to find a **Neural Lyapunov Surface** for the Collatz Conjecture (), I’ve demonstrated that even "chaotic" discrete rules can be modeled as a directed descent on a learned energy manifold.

---

## 💡 The Vision: Why Collatz?

Standard Gradient Descent (GD) works on smooth, continuous hills. But the real world—especially in **Quantized Neural Networks** and **Discrete Logic**—is "jumpy."

The Collatz Conjecture is the ultimate "jumpy" system. Traditionally viewed as chaotic, this project proves that we can find a potential function  where every step—including the massive "upward" jumps of the  rule—is actually moving "downhill" toward a global minimum (the 1-attractor).

### 🚀 Practical Implications for ML

* **Efficient Discrete Optimization:** Native integer-based gradients for hardware-constrained AI.
* **Escaping Local Minima:** Using  logic as a mathematical "escape hatch" to jump out of shallow valleys in a loss landscape.
* **Complexity Mapping:** Quantifying the "energy cost" of discrete states through bit-structure analysis.

---

## 🛤 The Research Journey

### Phase 1: The "56% Failure"

Initially, I attempted to train a model using raw integer values. The result was a **56% satisfaction rate**—barely better than a coin flip. The AI was blind to the logic: it saw  as a massive "error" because the value increased.

### Phase 2: The Bit-Structure Breakthrough

I realized the network needed to understand **Number Topology**, not just magnitude. I engineered a feature set focused on the binary and modular nature of the problem:

* **Modular Residue:**  to capture parity cycles.
* **Trailing Zeros:** Identifying "bit-shifts" (divisions by 2) that represent rapid descent.
* **Logarithmic Scaling:** Compressing the massive range of Collatz trajectories.

### Phase 3: Success

With these features, the model reached a **80%  satisfaction rate**. It successfully learned that . Even though the value increased, the **potential energy decreased**, proving the AI discovered the "progress" hidden in the jump.

---

## 🛠 Technical Implementation

### The Neural Lyapunov Loss

We define  as a scalar field. To satisfy the Lyapunov condition, we optimize for:



Where  is the Collatz map and  is a safety margin. The loss function penalizes any state where the next step results in an "energy increase."

### Feature Engineering

The model transforms a raw integer into a high-dimensional state vector:

1. ****: Global scale.
2. **Parity ()**: The primary branching factor.
3. **Local Geometry ()**: Identifying upcoming descent paths.
4. **Binary Density**: Count of trailing zeros in the bit-representation.

---

## 📊 Results & Analysis

| Range | Satisfaction Rate () | Avg. Energy Decrease |
| --- | --- | --- |
| **2 – 2,000 (Training)** | ~80.1% | 0.269 |
| **10,000 – 50,000 (Test)** | **87.06%** | **0.193** |

**Key Finding:** The model generalizes better on larger numbers it has *never seen before*. This suggests it has learned the **universal logic** of the Collatz rule rather than just memorizing paths.

---

## 🚦 Getting Started

### Prerequisites

* Python 3.8+
* TensorFlow 2.x
* NumPy & Matplotlib

### Installation

```bash
git clone https://github.com/your-username/collatz-lyapunov.git
cd collatz-lyapunov
pip install -r requirements.txt

```

### Run the Experiment

```bash
python collatz_nn.py

```

---

## 🔮 Future Work

* **Collatz-SGD:** Implementing a custom optimizer that uses the "parity-jump" logic to navigate non-convex loss surfaces in standard ML problems.
* **Proof of Monotonicity:** Investigating if a perfectly learned  could provide a computational approach to proving the original conjecture.
