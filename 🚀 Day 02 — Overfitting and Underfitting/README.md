Part of my **50-Day Machine Learning Journey**.

![Overfitting vs Underfitting]

## 🧠 What I Learned

After understanding the Machine Learning landscape on Day 1, today I learned about **Overfitting and Underfitting**.

A Machine Learning model should learn the important patterns in the training data and perform well on **new, unseen data**.

A model can fail in two opposite ways:

* **Underfitting** → the model is too simple and does not learn the pattern properly.
* **Overfitting** → the model is too complex and learns the training data too closely, including noise.

The main goal is to find a **good fit** where the model learns the important pattern without simply memorizing the training data.

---

# 📉 Underfitting

## What is Underfitting?

Underfitting happens when a model is **too simple to learn the important patterns** in the data.

Because the model has not learned enough, it performs poorly on both the training data and new data.

### Simple analogy

Imagine a student who barely studied for an exam.

The student performs poorly on:

* Practice questions
* The actual exam

The same idea applies to an underfitted Machine Learning model.

### Characteristics

* **Train score:** Low
* **Test score:** Low
* **Bias:** High

### Common causes

Underfitting can happen when:

* The model is too simple
* Important features are missing
* There is too little training
* Too much regularization is used

### How to fix underfitting

We can try:

* Using a more complex model
* Adding better features
* Training for longer
* Reducing excessive regularization

### Real-world examples

**House price prediction**

Using only the area of a house while ignoring:

* Location
* Number of rooms
* Age
* Facilities

may result in an overly simple model.

**Spam detection**

A model that only checks whether an email contains the word `"free"` is too simple to identify all types of spam.

---

# 📈 Overfitting

## What is Overfitting?

Overfitting happens when a model becomes **too complex** and learns the training data too closely.

It may learn both the actual pattern and the random noise in the training data.

As a result, it performs very well on training data but poorly on new data.

### Simple analogy

Imagine a student who memorizes every question from last year's exam.

They may score very well if the same questions appear again, but they may struggle when the questions are changed.

That is similar to overfitting.

### Characteristics

* **Train score:** Very high
* **Test score:** Much lower
* **Variance:** High

### Common causes

Overfitting can happen because of:

* A model that is too complex
* Too little training data
* Noisy data
* Excessive training
* Too many features relative to the amount of data

### How to fix overfitting

Common approaches include:

* Collecting more data
* Using a simpler model
* Regularization
* Cross-validation
* Early stopping
* Selecting useful features

### Real-world examples

**Face recognition**

A model trained on images of one person in one specific room may memorize the training conditions and perform poorly when lighting or background changes.

**Spam detection**

A model that memorizes exact training emails may fail when it receives new spam messages with different wording.

**Stock prediction**

A model may fit historical stock prices extremely well, including random fluctuations, but fail to predict future prices.

---

# ⚖️ Good Fit

The goal is not to make the model as simple or as complex as possible.

The goal is to find the right balance.

```text
Underfitting          Good Fit          Overfitting
     ↓                    ↓                  ↓
Too simple             Balanced          Too complex
Learns too little      Learns pattern     Learns pattern
                                          + noise
```

A good model should perform well on both training data and **unseen test data**.

---

# 🔎 How Can We Detect Underfitting and Overfitting?

One simple approach is to compare the model's **training score** and **test score**.

| Train Score | Test Score              | Interpretation |
| ----------- | ----------------------- | -------------- |
| Low         | Low                     | Underfitting   |
| High        | High and close to train | Good fit       |
| Very High   | Much lower              | Overfitting    |

For regression, `model.score()` commonly returns the **R² score**.

### R² Score

R² tells us how well the model explains the variation in the target.

* **1.0** → perfect prediction
* **0** → no better than predicting the average target value
* **Negative** → worse than that baseline

However, a high training R² alone does **not** mean the model is good. We also need to check how it performs on unseen test data.

---

# 🧪 Today's Experiment

To understand underfitting and overfitting practically, I used the **same dataset** and changed only the complexity of the model.

The dataset is based on a **noisy sine wave**.

The actual pattern is:

```python
np.sin(2 * np.pi * X)
```

Random noise is added to make the data more realistic.

The experiment compares different polynomial degrees:

| Model          | Degree | Expected Behavior |
| -------------- | -----: | ----------------- |
| Simple model   |      1 | Underfitting      |
| Balanced model |      3 | Good fit          |
| Complex model  |     15 | Overfitting       |

The important idea is:

> **The data stays the same. The model complexity changes.**

---

# 🔬 How the Experiment Works

### Creating the data

```python
rng = np.random.RandomState(42)
X = rng.rand(30, 1)
y = np.sin(2 * np.pi * X).ravel() + rng.normal(0, 0.2, 30)
```

The sine function creates the basic pattern.

Random noise is added to make the dataset less perfect.

`RandomState(42)` makes the random values reproducible, so the experiment gives consistent results when run again.

---

### Splitting the data

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
```

The data is divided into:

* **Training data** → used to teach the model
* **Testing data** → used to check how well the model performs on unseen data

With 30 data points and `test_size=0.3`:

* 21 points → training
* 9 points → testing

The test data acts like an **exam the model has never seen before**.

---

# 📉 Underfitting Experiment

For the underfitting experiment, I used:

```python
degree=1
```

A polynomial of degree 1 produces a **straight line**.

But the actual data follows a curved sine pattern.

Therefore, the straight line is too simple to represent the real pattern.

```text
Actual pattern → Curved
Model → Straight line
```

This demonstrates **underfitting**.

### File

`underfitting_demo.py`

---

# 📈 Overfitting Experiment

For the overfitting experiment, I used:

```python
degree=15
```

A degree-15 polynomial is much more flexible.

It can create a very complicated curve and may start following the random noise in the training data.

This demonstrates **overfitting**.

### File

`overfitting_demo.py`

---

# 🎯 Good Fit Experiment

A degree of approximately:

```python
degree=3
```

can provide a smoother curve for this particular dataset.

It is more flexible than a straight line but not as complex as the degree-15 model.

This demonstrates the idea of a **good fit**.

> The exact best degree depends on the dataset. Degree 3 is used here as a simple demonstration.

---

# 💻 Underfitting Code

### `underfitting_demo.py`

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

# Noisy curve data
rng = np.random.RandomState(42)
X = rng.rand(30, 1)
y = np.sin(2 * np.pi * X).ravel() + rng.normal(0, 0.2, 30)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Degree 1 means a straight line
model = make_pipeline(
    PolynomialFeatures(degree=1, include_bias=False),
    LinearRegression()
)

model.fit(X_train, y_train)

print("Train R2:", model.score(X_train, y_train))
print("Test R2: ", model.score(X_test, y_test))

# Plot
grid = np.linspace(0, 1, 200).reshape(-1, 1)

plt.plot(
    grid,
    np.sin(2 * np.pi * grid),
    "g--",
    label="True pattern"
)

plt.plot(
    grid,
    model.predict(grid),
    "r",
    label="Model (degree 1)"
)

plt.scatter(X_train, y_train, label="Train")
plt.scatter(X_test, y_test, marker="x", label="Test")

plt.title("Underfitting: line can't follow the curve")
plt.legend()
plt.show()
```

### What this code does

```text
Data
 ↓
Split into Train and Test
 ↓
Create degree-1 polynomial model
 ↓
Train model
 ↓
Calculate Train R²
 ↓
Calculate Test R²
 ↓
Plot the result
```

The straight line cannot follow the curved pattern well, so the model underfits.


<img width="783" height="436" alt="Screenshot 2026-10-07 at 18 03 46" src="https://github.com/user-attachments/assets/b99d237e-b241-43aa-967b-54657098604a" />
<img width="1146" height="703" alt="Screenshot 2026-10-07 at 18 02 57" src="https://github.com/user-attachments/assets/8356aad0-1d5a-4431-a708-8aba0bb6cc8a" />

---

# 💻 Overfitting Code

### `overfitting_demo.py`

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

# Same data as underfitting_demo.py
rng = np.random.RandomState(42)
X = rng.rand(30, 1)
y = np.sin(2 * np.pi * X).ravel() + rng.normal(0, 0.2, 30)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Degree 15 means a very flexible curve
model = make_pipeline(
    PolynomialFeatures(degree=15, include_bias=False),
    StandardScaler(),
    LinearRegression()
)

model.fit(X_train, y_train)

print("Train R2:", model.score(X_train, y_train))
print("Test R2: ", model.score(X_test, y_test))

# Plot
grid = np.linspace(0, 1, 200).reshape(-1, 1)

plt.plot(
    grid,
    np.sin(2 * np.pi * grid),
    "g--",
    label="True pattern"
)

plt.plot(
    grid,
    model.predict(grid),
    "r",
    label="Model (degree 15)"
)

plt.scatter(X_train, y_train, label="Train")
plt.scatter(X_test, y_test, marker="x", label="Test")

plt.ylim(-2, 2)

plt.title("Overfitting: curve chases the noise")
plt.legend()
plt.show()
```


<img width="783" height="424" alt="Screenshot 2026-10-07 at 18 04 37" src="https://github.com/user-attachments/assets/1e2e76fc-310d-43a9-9ad8-ea0484000147" />
<img width="1140" height="699" alt="Screenshot 2026-10-07 at 18 04 21" src="https://github.com/user-attachments/assets/fbd4cedb-b97d-4183-b06a-6dc3331f1b4b" />


### Why is `StandardScaler()` used here?

A degree-15 polynomial creates features such as:

```text
x
x²
x³
...
x¹⁵
```

These features can have very different numerical scales.

`StandardScaler()` rescales the features and helps make the calculations more numerically stable.

---

# 📊 My Experiment Results

After running both programs, record your actual results here:

| Degree |   Train R² |    Test R² | Interpretation |
| -----: | ---------: | ---------: | -------------- |
|      1 | Add result | Add result | Underfitting   |
|      3 | Add result | Add result | Good fit       |
|     15 | Add result | Add result | Overfitting    |

The important thing is not simply which model has the highest training score.

The important thing is **how well the model performs on unseen test data**.

---

# 🖼️ What the Graph Shows

In the experiment:

* **Green dashed line** → actual sine-wave pattern
* **Red line** → pattern learned by the model
* **Dots** → training points
* **X marks** → testing points

### Underfitting graph

The red line is too simple and cannot follow the curved pattern.

### Overfitting graph

The red curve becomes very complicated and starts following the noise in the training data.

### Good-fit graph

The model follows the main pattern without unnecessarily following every small fluctuation.

---

# 🌍 Real-World Understanding

The same problem occurs in real Machine Learning systems.

A model should not simply memorize historical data.

For example:

**House Price Prediction**

A model should learn general relationships between:

* Area
* Location
* Number of rooms
* Age
* Other relevant features

It should then be able to predict the price of a **new house**.

If the model only memorizes the houses in the training dataset, it may perform poorly on new houses.

That is why we care about **generalization**.

---

# 🧠 What I Learned Today

Today I learned that a Machine Learning model needs the right level of complexity.

### Underfitting

The model is **too simple**.

```text
Training → Poor
Testing → Poor
```

### Good Fit

The model learns the important pattern and generalizes well.

```text
Training → Good
Testing → Good and close to training
```

### Overfitting

The model is **too complex** and starts learning noise.

```text
Training → Very Good
Testing → Poor
```

The main goal is:

> **Build a model that performs well not only on the data it has seen, but also on new, unseen data.**

---

# 🔑 Key Takeaway

**Underfitting → Model learns too little.**

**Overfitting → Model learns too much, including noise.**

**Good fit → Model learns the important pattern and generalizes well.**

The most important lesson from today's experiment is:

> **A high training score does not automatically mean a good Machine Learning model. Always check how the model performs on unseen data.**

---

## 🛠️ Tools Used

* Python
* NumPy
* Scikit-learn
* Matplotlib

## 📁 Files

```text
Day-02/
│
├── README.md
├── underfitting_demo.py
├── overfitting_demo.py
│
└── images/
    └── overfitting-vs-underfitting.png
```

# ✅ Day 02 Complete

**Concept:** Overfitting vs Underfitting

**Main lesson:** Find the right balance between a model that is too simple and a model that is too complex.
