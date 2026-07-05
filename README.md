#  Formative 3: Probability Distributions, Bayesian Statistics & Gradient Descent
## Team Presentation Guide

**Team Members:**
- Oluwaloseyitowi Iji 
- Esther Mahoro
- Diane Ingabire
- Saad Byiringiro

  
**Presentation Date:** Tuesday, 7 July 2026  
**Time Slot:** 1:45pm   
**Total Duration:** 15 minutes

---

##  Presentation Overview

| Part | Topic | Speaker | Duration |
|------|-------|---------|----------|
| Part 1 | EM Algorithm & Probability Distributions | [] | 5 min |
| Part 2 | Bayesian Probability | [Towi] | 3 min |
| Part 3 & 4 | Gradient Descent (Manual & Code) | [Esther, Diane] | 7 min |

---

#  PART 1: Expectation-Maximization (EM) Algorithm
### Speaker: [Member Name] | Time: 0-5 minutes

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

### [MEMBER 1: INSERT YOUR CODE SNIPPETS HERE]

**Initialization:**
# [INSERT YOUR INITIALIZATION CODE]

E-Step Function:
<img width="620" height="86" alt="image" src="https://github.com/user-attachments/assets/493a994a-c87d-4b31-af93-963f48f9dab0" />

M-Step Function:
#### [INSERT YOUR M-STEP CODE]
Main EM Loop:
#### [INSERT YOUR MAIN LOOP CODE]

## Optimization Tracking Table
<img width="637" height="143" alt="image" src="https://github.com/user-attachments/assets/151c801b-ed52-4b16-8dd2-de33ec151dd6" />

# Visualizations
## Distribution Plot

<img width="218" height="126" alt="image" src="https://github.com/user-attachments/assets/7fb588b3-a0cc-41ae-9a09-3bf8267d8bd0" />

## Convergence Plot
<img width="218" height="126" alt="image" src="https://github.com/user-attachments/assets/e15e4228-58dc-44db-ab60-3d59b2c2dc4f" />


# PART 2: Bayesian Probability
- Apply Bayes' Theorem to calculate sentiment probabilities from IMDb movie reviews using basic Python only (no ML libraries).
## Positive Sentiment Keywords:
- **excellent** - Justification: strong praise; appears overwhelmingly in positive reviews
- **wonderful** - Justification: expresses delight; rare in negative reviews
- **brilliant** - Justification: high admiration for craft or performance


## Negative Sentiment Keywords:
- **worst** - Justification: a serious complaint; strongly negative
- **boring** - Justification: expresses disengagement; concentrated in negative reviews
- **waste** - Justification: signals regret ("waste of time"); almost always negative.

  
#  Bayes' Theorem
P(Positive|keyword) = P(keyword|Positive) × P(Positive) / P(keyword)

## Components:
- Prior: P(Positive): Base probability of positive reviews
- Likelihood: P(keyword|Positive): how often positive reviews use the keyword
- Marginal: P(keyword): how often the keyword appears overall
- Posterior: P(Positive|keyword) = Updated probability after seeing keyword

## Results
 
| Keyword | Prior | Likelihood | Marginal | Posterior |
|---------|-------|-----------|----------|-----------|
| excellent | 0.500 | 0.1147 | 0.0710 | **0.8074** |
| wonderful | 0.500 | 0.0903 | 0.0556 | **0.8122** |
| brilliant | 0.500 | 0.0635 | 0.0418 | **0.7601** |
| waste     | 0.500 | 0.0070 | 0.0507 | **0.0691** |
| worst     | 0.500 | 0.0164 | 0.0887 | **0.0927** |
| boring    | 0.500 | 0.0237 | 0.0610 | **0.1940** |

**Interpretation:** positive keywords push the posterior toward ~0.8, negative
keywords toward ~0.1, both starting from a neutral 0.5 prior. That swing away from 0.5 is Bayesian updating in action.

 
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

## Bayes' Theorem Implementation
```
def bayes(keyword):
    has_kw = df["review"].str.contains(rf"\b{keyword}\b", case=False, regex=True)

  likelihood = (has_kw & is_pos).sum() / prior         # P(keyword | Positive)
  marginal   = has_kw.sum()/len(df)                    # P(keyword)
  posterior  = (likelihood * prior) / marginal         # P(Positive | keyword)

  return prior, likelihood, marginal, posterior]
```
# Probability Results Tables
### Positive Keywords Analysis
<img width="617" height="184" alt="image" src="https://github.com/user-attachments/assets/05aa112e-fad4-43fe-98c5-6fa3f0bbeb54" />

### Negative Keywords Analysis
<img width="617" height="184" alt="image" src="https://github.com/user-attachments/assets/5bbf74a9-bad5-43dc-88d2-f9631d43a760" />


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
[MEMBER 4: INSERT YOUR MSE FUNCTION]

[INSERT YOUR MSE EQUATION CODE]

# Gradient Computation with SciPy
[INSERT YOUR SCIPY DERIVATIVE CODE]

## Training Loop
[INSERT YOUR GRADIENT DESCENT LOOP WITH VISIBLE STEPS]

## Sample Output
--- Iteration 1 ---
Predictions (ŷ):     [values]
Error (ŷ - y):       [values]
MSE:                 [value]
Gradient ∂J/∂m:      [values]
Gradient ∂J/∂b:      [values]
Updated m:           [values]
Updated b:           [values]

--- Iteration 2 ---
[Continue...]

## Results & Analysis
## Final Parameters
Final m = [values]
Final b = [values]
Final MSE = [value]
Final Predictions = [values]
Actual Values = [5, 6]

Key Observations:
- MSE consistently decreases with each iteration
- Parameters update in the opposite direction of gradients
- Algorithm successfully minimizes error
- Matrix operations maintained throughout (no scalar shortcuts)

## Trend Observations
[MEMBER 4: INSERT YOUR TREND ANALYSIS]

## Visualizations
### Parameter Trajectory
[INSERT YOUR PARAMETER PLOTTING CODE]
<img width="214" height="116" alt="image" src="https://github.com/user-attachments/assets/5c175864-5eaa-4e7d-876e-1fed135910fa" />

## Error Convergence
[MEMBER 3: INSERT YOUR MSE PLOTTING CODE]
<img width="214" height="116" alt="image" src="https://github.com/user-attachments/assets/5c175864-5eaa-4e7d-876e-1fed135910fa" />

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



