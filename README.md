#  Formative 3: Probability Distributions, Bayesian Statistics & Gradient Descent
## Team Presentation Guide

**Team Members:**
- Oluwaloseyitowi Iji 
- Esther Mahoro
- Diane Ingabire
- Saad Byiringiro

  
**Presentation Date:** Tuesday, 7 July 2026  
**Time Slot:** 1:45pm   
---

##  Presentation Overview

| Part | Topic | Speaker | Duration |
|------|-------|---------|----------|
| Part 1 | EM Algorithm & Probability Distributions | Towi, Diane | 5 min |
| Part 2 | Bayesian Probability | Towi | 3 min |
| Part 3 & 4 | Gradient Descent (Manual & Code) | Esther, Diane | 7 min |

---

#  PART 1: Expectation-Maximization (EM) Algorithm
### Speaker: Towi & Diane | Time: 0-5 minutes

##  Objective
Cluster unlabeled height data from two populations (Parents vs. Children) using a mixture of Gaussian distributions.

##  Critical Question: Why NOT Split at the Global Mean?

**Answer:** Splitting at the global mean is fundamentally flawed because:

1. **Truncation bias:** Cutting each group at the mean chops off the members that reach into the overlap, pushing the two estimated means artificially apart and shrinking both variances.
2. **Misclassification Risk:** Points in the overlapping region get forced into one side even though they could belong to either.
3. **Hard Assignments:** Forces binary classification instead of probabilistic reasoning
4. **Suboptimal Parameters:** Doesn't account for the actual shape of each distribution

**Solution:** Use EM algorithm with **soft assignments** (probabilistic responsibilities)


## Mathematical Foundation

### Gaussian Probability Density Function

P(x|μ,σ²) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))


### EM Algorithm Steps

**E-Step (Expectation):**
γ(i,k) = π_k * P(x_i|μ_k,σ_k²) / Σ_j π_j * P(x_i|μ_j,σ_j²)


**M-Step (Maximization):**
μ_k = Σ_i γ(i,k) * x_i / Σ_i γ(i,k)
σ_k² = Σ_i γ(i,k) * (x_i - μ_k)² / Σ_i γ(i,k)
π_k = Σ_i γ(i,k) / N

**Convergence:** stop when the log-likelihood changes by less than a small tolerance.


##  Implementation Code
```
fathers = df.drop_duplicates("family")["father"].dropna().to_numpy()
children = df["childHeight"].dropna().to_numpy()
x = np.concatenate([fathers, children])   
```
```
def gaussian(x, mu, var):
    """Height of the bell curve (mean mu, variance var) at point x."""
    return np.exp(-(x - mu) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)
```
```
def log_likelihood(x, mu1, mu2, var1, var2, pi1, pi2):
    """Total log-probability of the data under the current mixture.
    This is the single score EM pushes UP every iteration."""
    mixed = pi1 * gaussian(x, mu1, var1) + pi2 * gaussian(x, mu2, var2)
    return np.sum(np.log(mixed))
```
```
global_mean = x.mean()
low = x[x < global_mean]     # everyone below the average -> pile 1
high = x[x >= global_mean]  # everyone above the average -> pile 2
```
```
mu1, mu2 = low.mean(), high.mean()
var1, var2 = low.var(), high.var()
pi1, pi2 = len(low) / N, len(high) / N
```
**E-step & M-step**:
```
def em(x, mu1, mu2, var1, var2, pi1, pi2, max_iters=500, tol=1e-6, verbose=True):
    history = []
 
    def record(it, ll):
        history.append((it, mu1, mu2, var1, var2, pi1, pi2, ll))
 
    ll = log_likelihood(x, mu1, mu2, var1, var2, pi1, pi2)
    record(0, ll)
    if verbose:
        print(f"{'it':>3}{'mu1':>9}{'mu2':>9}{'var1':>9}{'var2':>9}{'pi1':>7}{'pi2':>7}{'LL':>11}")
        print(f"{0:>3}{mu1:9.3f}{mu2:9.3f}{var1:9.3f}{var2:9.3f}{pi1:7.3f}{pi2:7.3f}{ll:11.2f}")
 
    prev = ll
    for it in range(1, max_iters + 1):
        # ---- E-step: responsibilities (soft assignment) ----
        r1 = pi1 * gaussian(x, mu1, var1)    # unnormalized pull toward comp 1
        r2 = pi2 * gaussian(x, mu2, var2)    # unnormalized pull toward comp 2
        total = r1 + r2
        g1 = r1 / total                      # P(comp1 | height)
        g2 = r2 / total                      # P(comp2 | height)
 
        # ---- M-step: re-estimate each curve using the soft assignments ----
        N1, N2 = g1.sum(), g2.sum()          # soft count
        mu1 = (g1 * x).sum() / N1
        mu2 = (g2 * x).sum() / N2
        var1 = (g1 * (x - mu1) ** 2).sum() / N1
        var2 = (g2 * (x - mu2) ** 2).sum() / N2
        pi1, pi2 = N1 / N, N2 / N
 
        ll = log_likelihood(x, mu1, mu2, var1, var2, pi1, pi2)
        record(it, ll)
        if verbose and it <= 2:              # print the first two updates for the table
            print(f"{it:>3}{mu1:9.3f}{mu2:9.3f}{var1:9.3f}{var2:9.3f}{pi1:7.3f}{pi2:7.3f}{ll:11.2f}")
 
        if abs(ll - prev) < tol:
            break
        prev = ll
 
    params = dict(mu1=mu1, mu2=mu2, var1=var1, var2=var2, pi1=pi1, pi2=pi2)
    return params, history, it

```
```
params, history, n_iters = em(x, mu1, mu2, var1, var2, pi1, pi2)
```
**Output- Optimization tracking Table**:

<img width="230" height="47" alt="Screenshot 2026-07-07 131948" src="https://github.com/user-attachments/assets/0a100ed9-f6e1-455f-b861-69d69c0c415f" />

# Visualizations
## Distribution Plot

<img width="338" height="203" alt="Screenshot 2026-07-07 132324" src="https://github.com/user-attachments/assets/4c872da1-b2ae-45e7-bcee-9d5aab1369b4" />


## Convergence Plot

<img width="341" height="203" alt="Screenshot 2026-07-07 132335" src="https://github.com/user-attachments/assets/6d7bf587-5a04-46d3-a9ca-4b6ff7398c67" />



# PART 2: Bayesian Probability
- Apply Bayes' Theorem to calculate sentiment probabilities from IMDb movie reviews using basic Python only (no ML libraries).
## Positive Sentiment Keywords:
- **excellent** - Justification: strong praise; appears overwhelmingly in positive reviews
- **wonderful** - Justification: expresses delight; rare in negative reviews
- **brilliant** - Justification: high admiration for craft or performance


## Negative Sentiment Keywords:
- **worst** - Justification: a serious complaint; strongly negative
- **bad** - Justification: expresses disengagement; concentrated in negative reviews
- **waste** - Justification: signals regret ("waste of time"); almost always negative.

  
#  Bayes' Theorem
P(Positive|keyword) = P(keyword|Positive) × P(Positive) / P(keyword)

## Components:
- Prior: P(Positive): Base probability of positive reviews
- Likelihood: P(keyword|Positive): how often positive reviews use the keyword
- Marginal: P(keyword): how often the keyword appears overall
- Posterior: P(Positive|keyword) = Updated probability after seeing keyword

 
# Implementation Code

#### 
```
df = pd.read_csv("IMDB Dataset.csv", delimiter=",")
df.head()]
```

## Counting Keywords
```
positive_keywords = ["excellent", "wonderful", "brilliant"]
negative_keywords = ["waste", "worst", "boring"]

total = len(df)
is_pos = df["sentiment"] == "positive"
positives = df[is_pos]
is_neg = df["sentiment"] == "negative"
negatives = df[is_neg]
positives

prior = len(positives) / total
prior
```
## Word justification:
```
from collections import Counter

pos_counts, neg_counts = Counter(), Counter()
for text, sentiment in zip(df["review"], df["sentiment"]):
    words = set(text.lower().split())
    (pos_counts if sentiment == "positive" else neg_counts).update(words)

leans = []
for w in set(pos_counts) | set(neg_counts):
    p, n = pos_counts[w], neg_counts[w]
    if p + n < 1500 or len(w) < 4:      # frequent, real words only
        continue
    leans.append((w, p / (p + n)))      # pos_share: 1.0 = very positive, 0.0 = very negative

leans.sort(key=lambda t: t[1], reverse=True)
print("Most POSITIVE-leaning:", [w for w, _ in leans[:8]])
print("Most NEGATIVE-leaning:", [w for w, _ in leans[-8:]])
```
**Output**:
<img width="402" height="27" alt="image" src="https://github.com/user-attachments/assets/28496a6b-5c8b-4f83-867d-2c0828e2c42d" />

## Bayes' Theorem Implementation
```
def bayes(keyword):
    has_kw = df["review"].str.contains(rf"\b{keyword}\b", case=False, regex=True)

  likelihood = (has_kw & is_pos).sum() / prior         # P(keyword | Positive)
  marginal   = has_kw.sum()/len(df)                    # P(keyword)
  posterior  = (likelihood * prior) / marginal         # P(Positive | keyword)

  return prior, likelihood, marginal, posterior]
```


**Interpretation:** positive keywords push the posterior toward ~0.8, negative
keywords toward ~0.1, both starting from a neutral 0.5 prior. That swing away from 0.5 is Bayesian updating in action.

# Probability Results Tables
### Positive Keywords Analysis

<img width="258" height="114" alt="Screenshot 2026-07-07 125227" src="https://github.com/user-attachments/assets/48edaf9c-ca1f-4abf-8140-bb15d1119863" />


### Negative Keywords Analysis

<img width="255" height="110" alt="Screenshot 2026-07-07 125237" src="https://github.com/user-attachments/assets/7ed352f6-c3b1-4f51-b7ce-a10f5308442c" />



##  PART 3 & 4: Gradient Descent

Implement gradient descent for linear regression using matrix operations, both manually (Part 3) and in Python with SciPy (Part 4).

## Problem Setup

## Linear Equation:
y = m₁x₁ + m₂x₂ + b

#### Given Parameters:
- Initial m = [-1, 2]
- Initial b = [1, 1] (Note: Treated as vector, NOT scalar)
- Data X = [[1, 3], [4, 10]]
- Data y = [5, 6]
- Learning Rate (α) = 0.001
- Iterations = 4
### Part 3: Manual Calculation
## Cost Function: Mean Squared Error (MSE)
J(m,b) = (1/n) × Σ(ŷ - y)²
where ŷ = X·m + b

## Chain Rule Derivatives

<img width="183" height="251" alt="image" src="https://github.com/user-attachments/assets/c2b7f3b3-62f0-42a9-bc94-437d22f09767" /> <img width="186" height="275" alt="image" src="https://github.com/user-attachments/assets/dda88f23-75f8-42db-9101-4c8571403103" />



## Derivative with respect to m:
∂J/∂m = (2/n) × Xᵀ × (ŷ - y)

## Derivative with respect to b:
∂J/∂b = (2/n) × (ŷ - y)
Manual Iteration Results:
[MEMBER 3: INSERT YOUR STEP-BY-STEP CALCULATIONS]

ŷ = [values]
Error = [values]
∂J/∂m = [values]
∂J/∂b = [values]
m_new = [values]
b_new = [values]


##  Part 4: Python Implementation

## Setup

import numpy as np
from scipy.optimize import approx_fprime
import matplotlib.pyplot as plt

### Initial parameters
m = np.array([-1.0, 2.0])
b = np.array([1.0, 1.0])  # Vector, not scalar!

### Data
X = np.array([[1, 3], 
              [4, 10]])
y = np.array([5, 6])

### Hyperparameters
learning_rate = 0.001
iterations = [NUMBER]

# MSE Equation Function
<img width="644" height="65" alt="image" src="https://github.com/user-attachments/assets/5fb3765f-a88b-48c6-b6c5-2b64fde95550" />

# Gradient Computation with SciPy
<img width="640" height="218" alt="image" src="https://github.com/user-attachments/assets/95014e77-ae63-4b09-98ef-f96eb85d3a91" />


## Training Loop
<img width="781" height="356" alt="image" src="https://github.com/user-attachments/assets/bf06846c-0fe0-44b6-bac6-d69122b0dbaf" />

## Sample Output
<img width="781" height="379" alt="image" src="https://github.com/user-attachments/assets/30870173-0317-4990-a649-e2ee24e0de33" />


Key Observations:
- MSE consistently decreases with each iteration
- Parameters update in the opposite direction of gradients
- Algorithm successfully minimizes error
- Matrix operations maintained throughout (no scalar shortcuts)

## Trend Observations
[MEMBER 4: INSERT YOUR TREND ANALYSIS]

## Visualizations
### Parameter Trajectory
<img width="544" height="304" alt="image" src="https://github.com/user-attachments/assets/b17dc1e8-4124-4774-b883-d16364b621db" />


## Error Convergence
<img width="553" height="319" alt="image" src="https://github.com/user-attachments/assets/cbcc654a-d76f-4981-87ce-996fed677365" />


## Repository Structure

formative-3/
├── README.md                          # This file
├── notebook.ipynb                     # Complete Jupyter Notebook
├── Part3_Manual_Calculations.pdf      # Handwritten calculations
├── Contribution_Proof.pdf             # Team contributions
├── images/
│   ├── em_distributions.png
│   ├── em_convergence.png
│   ├── bayesian_probs.png
│   ├── gd_parameters.png
│   └── gd_mse.png
└── data/
    ├── heights.csv
    └── imdb_reviews.txt

## Technical Setup

### Clone repository
git clone [[repo-url]](https://github.com/teniolaiji/MLMath_Formative3_Group7)
cd formative-3

### Install dependencies
pip install numpy scipy matplotlib

### Run notebook
jupyter notebook Assignment.ipynb



