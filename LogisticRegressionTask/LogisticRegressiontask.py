# %% [markdown]
# <a href="https://colab.research.google.com/github/Aneeshshastri/Epoch_Spring_Camp/blob/main/LogisticRegression.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Logistic Regression from Scratch
# 
# We’ll build a logistic regression model step by step using synthetic mouse data.  
# The dataset includes **Weight (g)** and **Height (cm)**, and a binary target **Obese (1) or Healthy (0)**.  

# %% [markdown]
# ## 1. What is Logistic Regression?
# Logistic regression is a classification algorithm.  
# It predicts the probability that an input belongs to a certain class (e.g., obese vs. not obese).  
# 
# The model equation is:
# 
# $$
# P(y=1|x) = \sigma(w^T x + b)
# $$
# 
# where:
# - $w$ = weights (learnable parameters for each feature)
# - $b$ = bias (learnable parameter)
# - $\sigma(z) = \frac{1}{1+e^{-z}}$ = sigmoid function
# 
# ---

# %% [markdown]
# 
# ## 2. Likelihood as Conditional Probability
# Likelihood answers the question: *“Given my parameters $w$, how probable is it that I observe this dataset?”*
# 
# For binary classification:
# - Each label $y_i$ is either 0 or 1.
# - The probability of seeing $y_i$ given input $x_i$ is:
# 
# $$
# P(y_i|x_i; w) = p_i^{y_i} (1-p_i)^{(1-y_i)}
# $$
# 
# where $p_i = \sigma(w^T x_i)$.
# 
# The **likelihood of the whole dataset** is the product of all these probabilities:
# 
# $$
# L(w) = \prod_{i=1}^n P(y_i|x_i; w)
# $$
# 
# ---
# 

# %% [markdown]
# 
# ## 3. Log-Likelihood and Binary Cross Entropy
# Multiplying many probabilities can get very small, so we take the log:
# 
# $$
# \ell(w) = \sum_{i=1}^n \Big[ y_i \log(p_i) + (1-y_i)\log(1-p_i) \Big]
# $$
# 
# - This is called the **log-likelihood**.
# - **Maximum Likelihood Estimation (MLE)** means: find $w$ that maximizes $\ell(w)$.
# 
# Instead of maximizing, we minimize the **negative log-likelihood**:
# 
# $$
# J(w) = -\ell(w)
# $$
# 
# This is exactly the **Binary Cross Entropy Loss**:
# 
# $$
# J(w) = -\frac{1}{n} \sum_{i=1}^n \Big[ y_i \log(p_i) + (1-y_i)\log(1-p_i) \Big]
# $$
# 
# So: **minimizing binary cross entropy = maximizing likelihood.**
# 
# ---

# %% [markdown]
# 
# ## 4. Gradient Descent and the Update Rule
# To minimize the loss, we compute the gradient (derivative of loss wrt weights):
# 
# $$
# \nabla J(w) = \frac{1}{n} X^T (h - y)
# $$
# 
# This tells us how to adjust weights to reduce loss.
# 
# Update step:
# 
# $$
# w \leftarrow w - \eta \cdot \nabla J(w)
# $$
# 
# - $\eta$ = learning rate (step size).
# - Each update moves weights closer to the values that maximize likelihood.
# - Iteratively applying this update is how logistic regression learns.
# 
# ---

# %%
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set a seed for reproducibility
np.random.seed(42)

# %%
# Generate synthetic Mouse Data
n_samples = 100
weight = np.random.normal(30, 5, n_samples)
height = np.random.normal(10, 2, n_samples)

noise = np.random.normal(0, 2, n_samples)
condition = (weight + height + noise) > 42
target = condition.astype(int)


# %%
# Create DataFrame
df = pd.DataFrame({
    'Weight': weight,
    'Height': height,
    'Obese': target
})

# Quick look at the first 5 mice
print(df.head())

# %%
# Visualize the clusters
plt.scatter(df['Weight'], df['Height'], c=df['Obese'], cmap='bwr', edgecolors='k')
plt.xlabel('Weight (g)')
plt.ylabel('Height (cm)')
plt.title('Mouse Obesity: Red=Obese, Blue=Healthy')
plt.show()

# %% [markdown]
# ## Preprocessing Data
# 
# Before training, we need to:
# 1. Extract features (Weight, Height) and target (Obese).
# 2. Standardize features so they’re on the same scale.
# 3. Add a bias term (column of ones) for the intercept.

# %%
def preprocess_data(df):
    # Extract features and target
    X = df[['Weight', 'Height']].values
    y = df['Obese'].values.reshape(-1, 1)

    # TODO: Standardize the features ( (X - mean) / std )
    X_scaled =  X-X.mean(axis=0)/X.std(axis=0)

    # TODO: Add a column of ones to X for the bias term (np.hstack)
    X_final = np.hstack((np.ones((X_scaled.shape[0], 1)), X_scaled))

    return X_final, y

# Run the function
X_train, y_train = preprocess_data(df)
print(X_train.shape,y_train.shape)

# %% [markdown]
# ## Logistic Regression Implementation
# 
# We’ll implement logistic regression **from scratch**:
# - Sigmoid function
# - Binary Cross Entropy loss
# - Gradient descent for weight updates
# 

# %%
class LogisticRegressionScratch:
    def __init__(self, lr=1e-2, iters=100):
        self.lr = lr
        self.iters = iters
        self.weights = None
        self.losses = []

    def sigmoid(self, z):

        return 1 / (1 + np.exp(-z))

    def compute_loss(self, y, h):

        return (-1/n_samples)*((y*np.log(h))+(1-y)*(np.log(1-h))).sum()

    def fit(self, X, y):
        # Initialize weights (size = number of features)
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1))

        for i in range(self.iters):
            # 1. Forward Pass: Calculate z and h
            z=np.dot(X,self.weights)
            # 2. Compute Loss and append to self.losses
            h=self.sigmoid(z)
            loss=self.compute_loss(y,h)
            self.losses.append(loss)
            # 3. Calculate Gradient (dw)
            dw=(1/n_samples)*X.T@(h-y)
            # 4. Update Weights
            self.weights-=self.lr*dw


    def predict(self, X):
        # TODO: Return 1 if sigmoid(Xw) > 0.5 else 0
        return  self.sigmoid(np.dot(X,self.weights))>0.5
    def logit(self,X):
        return np.dot(X,self.weights)


# %% [markdown]
# ## Training the Model
# 
# Now we’ll train our logistic regression model on the mouse dataset.  
# We’ll also plot the **loss history** to check if the model is learning.
# 

# %%
# 1. Complete the TODOs above.
# 2. Train the model using X_train, y_train.
# 3. Plot the 'Loss History' to see if it decreases over time.

# %%
# Train the model
model = LogisticRegressionScratch(lr=1e-2, iters=40_000)
model.fit(X_train, y_train)

# Plot loss history
plt.plot(model.losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss History")
plt.show()

# Predictions
y_pred = model.predict(X_train)
#plt.show(
# Accuracy
accuracy = (y_pred == y_train).mean()

plt.scatter(df['Weight'], df['Height'], c=df['Obese'], cmap='bwr', edgecolors='k')
plt.figure()
plt.scatter(df['Weight'], df['Height'], c=y_pred, cmap='bwr', edgecolors='k')
print("Training Accuracy:", accuracy)

#print(y_pred)


