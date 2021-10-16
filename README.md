# Machine Learning Definition
- Arthur Samuel (1959). Field of study that gives computers the ability to learn without being explicitly programmed.
- Tom Mitchell (1998) Well-posed Learning Problem: A computer is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E.
# Machine Learning Algorithms
## Supervised Learning
In supervised learning, we are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and the output.
### Regression
In a regression problem, we are trying to predict results within a continuous output, meaning that we are trying to map input variables to some continuous function.
- **Linear Regression or Univariate(one variable) Linear Regression**
**Hypothesis:** $h_{\theta}(x)=\theta_{0}+\theta_{1}x$
### Classification
In a classification problem, we are instead trying to predict results in a discrete output. In other words, we are trying to map input variables into discrete categories.
## Unsupervised Learning
Unsupervised learning allows us to approach problems with little or no idea what our results should look like. We can derive structure from data where we don't necessarily know the effect of the variables.
### Clustering
We can derive this structure by clustering the data based on relationships among the variables in the data.
### Dimensionality Reduction
# Others: Reinforcement Learning, Recommender Systems

# Loss Function
The loss function (or error) is for a single training example, while the cost function is over the entire training set.
# Cost Function
We can measure the accuracy of our hypothesis function by using a cost function. The goal is to minimize the cost function.
## Squared Error Function or Mean Squared Error(MSE)
$J(\theta_{0},\theta_{1})=\frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^i)-y^i)^2$
$x^i$->input variable
$y^i$->output variable
$m$->training set
$h_{\theta}(x^i)-y^i$->difference between the predicted value and the actual value
# Gradient Descent
repeat until convergence { $\theta_{j} := \theta_{j}-\alpha\frac{\partial}{\partial\theta_{j}}J(\theta_{0},\theta_{1})$     (for $j=0$ and $j=1$) }
$:=$->assignment mark
$\alpha$->learning rate
## Local Optimum
## Global Optimum

```latex
\begin{equation*}
B = 
\begin{bmatrix}
1402 & 191 \\
1371 & 821 \\
949 & 1437 \\
147 & 1448 \\
\end{bmatrix}
\end{equation*}
```
