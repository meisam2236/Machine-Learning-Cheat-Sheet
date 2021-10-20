- [Machine Learning Definition](#machine-learning-definition)
- [Machine Learning Algorithms](#machine-learning-algorithms)
  - [Supervised Learning](#supervised-learning)
    - [Regression](#regression)
    - [Classification](#classification)
  - [Unsupervised Learning](#unsupervised-learning)
    - [Clustering](#clustering)
    - [Dimensionality Reduction](#dimensionality-reduction)
- [Others: Reinforcement Learning, Recommender Systems, Association Rules](#others-reinforcement-learning-recommender-systems-association-rules)
- [Noise](#noise)
  - [Teacher noise](#teacher-noise)
- [Cost Function](#cost-function)
  - [Loss(Error) Function](#losserror-function)
  - [Squared Error Function or Mean Squared Error(MSE)](#squared-error-function-or-mean-squared-errormse)
  - [(Batch)Gradient Descent](#batchgradient-descent)
    - [Local Optimum](#local-optimum)
    - [Global Optimum](#global-optimum)
  - [Interpolation](#interpolation)
- [Multivariate Linear Regression(multiple features)](#multivariate-linear-regressionmultiple-features)
  - [Cost Function](#cost-function-1)
  - [Gradient Descent](#gradient-descent)
  - [Feature Scaling](#feature-scaling)
    - [Mean Normalization](#mean-normalization)
    - [How to make sure gradient descent is working correctly](#how-to-make-sure-gradient-descent-is-working-correctly)
  - [Creating new Features](#creating-new-features)
- [Fitting approach](#fitting-approach)
- [Polynomial Regression](#polynomial-regression)
  - [Extrapolation](#extrapolation)
- [Normal Equation](#normal-equation)
  - [What if $X^TX$ is non-invertible](#what-if-xtx-is-non-invertible)
# Machine Learning Definition
- Arthur Samuel (1959). Field of study that gives computers the ability to learn without being explicitly programmed.
- Tom Mitchell (1998) Well-posed Learning Problem: A computer is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E.

A machine learning model is a file that has been trained to recognize certain types of patterns. The model may be **predictive** to make predictions in the future, or **descriptive** to gain knowledge from data, or both.
# Machine Learning Algorithms
## Supervised Learning
In supervised learning, the aim is to learn a mapping from the input to an output whose correct values are provided by a supervisor. 
or in other word, in supervised learning, we are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and the output.
### Regression
In a regression problem, we are trying to predict results within a continuous output, meaning that we are trying to map input variables to some continuous function.
- **Linear Regression or Univariate(one variable) Linear Regression**
**Hypothesis:** $h_{\theta}(x)=\theta_{0}+\theta_{1}x$
or by other form: $y=wx+w_0$
### Classification
In a classification problem, we are instead trying to predict results in a discrete output. In other words, we are trying to map input variables into discrete categories.
In engineering, classification is called **pattern recognition**.

**Empirical error** is the proportion of training instances where predictions of h do not match the required values given in X. The empirical error is also sometimes called **the generalization error**. The reason is that actually, in most problems, we don't have access to the whole domain X of inputs, but only our training subset S. We want to generalize based on S, also called inductive learning.

**False positive** is a result that indicates a given condition exists when it does not.
**False negative** is a result that indicates a given condition does not exist when it exists.

**Vapnik-Chervonenkis Dimension:**
The maximum number of points that can be shattered by H is called the Vapnik-Chervonenkis (VC) dimension of H, is denoted as VC(H), and measures the capacity of H.
With high probability$(1-\eta)$, Vapnik showed:
$TestError \leq TrainError+\sqrt{\frac{H \log{(\frac{2m}{H})+H-log{\frac{\eta}{4}}}}{m}}$
H->VC dimension of the classifier
The equation above shows if H is very small or m is very large, the additional bounding term will be small. So low VC dimension compared to data will suggest that training and test error will be quite similar!
E.g. If you have a line and want to seperate n lines, VC dimension will be $2^n$.

**Probably approximately correct(PAC):**
The goal of PAC is with high probability(probably), the selected hypothesis will have lower error(approximately correct).
In the PAC model, we specify two small parameters, $\varepsilon$ and $\delta$, and require that with probability at least $(1-\delta)$ a system learn a concept with error at most $\varepsilon$.
$\varepsilon$ gives an upper bound on the error in accuracy with which h approximated(accuracy: $1-\varepsilon$)
E.g. $P(C XOR h) \leq \varepsilon$
A hypothesis is said to be approximately correct, if the error is less than or equal to $\varepsilon$, where $0 \leq \varepsilon \leq \frac{1}{2}$
$\delta$ gives the probability of failure in achieving this accuracy(confidence: $1-\delta$)
E.g. $P(Error(h)\leq \varepsilon)>1-\delta$
or $P(P(C XOR h) \leq \varepsilon)>1-\delta$
## Unsupervised Learning
In unsupervised learning, there is no such supervisor and we have only input data.
It allows us to approach problems with little or no idea what our results should look like. We can derive structure from data where we don't necessarily know the effect of the variables.
### Clustering
We can derive this structure by clustering the data based on relationships among the variables in the data.
### Dimensionality Reduction
# Others: Reinforcement Learning, Recommender Systems, Association Rules
In some applications, the output is a sequence of actions. In such case a single action is not important; what is important is the policy that is the sequence of correct actions to reach the goal. There is no such thing as the best action in any intermediate state; an action is good if it is part of a good policy. In such a case, the machine learning program should be able to assess the goodness of policies and learn from past good action sequences to be able to generate a policy.

# Noise
Noise is any unwanted anomaly in the data and due to noise, the class may be more difficult to learn and zero error may be infeasible with a simple hypothesis class.
## Teacher noise
There may be errors in labeling the data points, which may relabel positive instances as negative and vice verca. This is called teacher noise.

# Cost Function
We can measure the accuracy of our hypothesis function by using a cost function. The goal is to minimize the cost function.
## Loss(Error) Function
The loss function (or error) is for a single training example, while the cost function is over the entire training set.
## Squared Error Function or Mean Squared Error(MSE)
$$J(\theta_{0},\theta_{1})=\frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^i)-y^i)^2$$
$x^i$->input variable
$y^i$->output variable
$m$->training set
$h_{\theta}(x^i)-y^i$->difference between the predicted value and the actual value
## (Batch)Gradient Descent
Batch in it indecates that each step of gradient descent uses all the training examples.
repeat until convergence { $\theta_{j} := \theta_{j}-\alpha\frac{\partial}{\partial\theta_{j}}J(\theta_{0},\theta_{1})$     (for $j=0$ and $j=1$) }
simplified { $\theta_{0} := \theta_{0}-\alpha\frac{1}{m}\sum{i=1}{m} (h_{\theta}(x^i)-y^i)$     $\theta_{1} := \theta_{1}-\alpha\frac{1}{m}\sum{i=1}{m} (h_{\theta}(x^i)-y^i).x^i$}
$:=$->assignment mark
$\alpha$->learning rate
$\frac{\partial}{\partial\theta_{j}}J(\theta_{0},\theta_{1})$->derivative term
**Tip:** Simpultaneously update $\theta_{0}$ and $\theta_{1}$
### Local Optimum
A solution that is optimal within a neighboring set of candidate solutions.
### Global Optimum
A solution that is optimal through entire set of candidate solutions.

When gradient descent converged to a local minimum, it will be unchanged because of the slope(which is zero) even with the learning rate $\alpha$ fixed(That's because derivative term will be smaller as you approach to the local minimum).
Gradient Descent for Linear Regression always leads to Convex Function(Bowl-shaped Function). So we just have one local optimum that is global optimum!
## Interpolation
In regression, if there is no noise, the task is interpolation.
# Multivariate Linear Regression(multiple features)
$n$->number of features
$x^{(i)}$->input (features) of $i^{th}$ training example
$x_{j}^{(i)}$->value of feature $j$ in $i^{th}$ training example

**Hypothesis:** $h_{\theta}(x)=\theta_{0}+\theta_{1}x_{1}+\theta_{2}x_{2}+...+\theta_{n}x_{n}$
For convenience of notation, define $x_{0}=1$:
$$
x=
\begin{bmatrix}
x_{0} \\
x_{1} \\
x_{2} \\
... \\
x_{n}
\end{bmatrix}
\in {\rm I\!R}^{n+1}
\ \ \ \ \ \ \ \ \ \ \ \ 
\theta=
\begin{bmatrix}
\theta_{0} \\
\theta_{1} \\
\theta_{2} \\
... \\
\theta_{n}
\end{bmatrix}
\in {\rm I\!R}^{n+1}
$$
**Hypothesis will be:** $h_{\theta}(x)=\theta_{0}x_{0}+\theta_{1}x_{1}+\theta_{2}x_{2}+...+\theta_{n}x_{n}=\theta^{T}x$
## Cost Function
$J(\theta_{0},\theta_{1},...,\theta_{n})=J(\theta)=\frac{1}{2m}\sum_{i=1}^{m}(h_{\theta}(x^{i})-y^{i})^{2}$
## Gradient Descent
Repeat {
$\theta_{j}:=\theta_{j}-\alpha\frac{\partial}{\partial\theta_{j}}J(\theta)$
} (simultaneously update for every $j=0,...,n$)
simplified {
$\theta_{0}:=\theta_{0}-\alpha\frac{1}{m}\sum_{i=1}^{m}(h_{\theta}(x^i)-y^i)x_{0}^i$
$\theta_{1}:=\theta_{1}-\alpha\frac{1}{m}\sum_{i=1}^{m}(h_{\theta}(x^i)-y^i)x_{1}^i$
$\theta_{2}:=\theta_{2}-\alpha\frac{1}{m}\sum_{i=1}^{m}(h_{\theta}(x^i)-y^i)x_{2}^i$
...
}
## Feature Scaling
The idea is to make sure features are on a similar scale(Thus makes the gradient descending convergence much faster!).
**E.g.**
1. $x_{1}$ = size (0-2000 square feet) -> $x_{1}=\frac{size(feet^{2})}{2000}$ 
2. $x_{2}$ = number of bedrooms (1-5) -> $x_{2}=\frac{number of bedrooms}{5}$

So we end up with $0 \leqslant x_{1} \leqslant1$ and $0 \leqslant x_{2} \leqslant1$
More generally what we often want to do is to get every feature into approximately a $-1 \leqslant x_{i} \leqslant1$ range.
### Mean Normalization
Replace $x_{i}$ with $x_{i}-\mu_{i}$ to make features have approximately zero mean(Do not apply to $x_{0}=1$).
**E.g.**
1. $x_{1}=\frac{size-1000}{2000}$ Assume the average size is 1000
2. $x_{2}=\frac{number of bedrooms-2}{5}$ Assume average number of bedrooms is 2

And we end up with $-0.5 \leqslant x_{1} \leqslant -0.5, -0.5 \leqslant x_{2} \leqslant 0.5$

More general formula will be $x_{1}=\frac{x_{1}-\mu_{1}}{S_{1}}$
$\mu_{1}$->average value of $x_{1}$ in training set
$S_{1}$->range of values of $x_{1}$(maximum value-minimum value)
The $S_{1}$ could be standard deviation(better results):
$\sigma=\sqrt{\frac{\sum(x_{i}-\mu)^{2}}{N}}$
$\sigma$->standard deviation
$N$->size of values
$\mu$->mean
### How to make sure gradient descent is working correctly
You can draw chart of y=Cost Function, x=Number of Iterations($J(\theta)$ should decrease after every iteration) and when it converge, you can stop program.
You can come up with some automatic convergence test, like:
Declare convergence if $J(\theta)$ decreases by less than $10^{-3}$ in one iteration.
The threshold picking in automatic convergence test is quite hard! So it's better to use the chart.
**Debugging**
If $J(\theta)$ is increasing, it's a clear sign that you should use smaller $\alpha$.
If $J(\theta)$ goes up and down repeatedly, you should use smaller $\alpha$.
If $\alpha$ is too small, gradient descent can be slow to converge.

To choose $\alpha$, try 
$$..., \ \ \ 0.001, \ \ \ 0.01, \ \ \ 0.1, \ \ \ 1, \ \ \ ...$$
Then try to do $3\times$ more or less to find best fitted!

## Creating new Features
Sometimes, depending on your problem, you can create better input to deal with. For example, if you have frontage and depth of a house to calculate the pricing, you can come up with better input which is area, by multiplying frontage by depth!

# Fitting approach
With a complex model, one can make a perfect fit to the data and attain zero errro. Another possibility is to keep the model simple and allow some error.
Using the simple one makes more sense:
1. It is a simple model to use.
2. It is a simple model to train and has fewer parameters.
3. It is a simple model to explain.
4. It has less variance and is less affected by single instances.

A simple model have less variance and more bias. Finding the optimal model corresponds to minimizing bothe the bias and the variance.
Remember a simple(but not too simple) model would generalize better than a complex model.
# Polynomial Regression
In cases where the linear model is too restrictive, we can use polynomial regression.
With polynomial, we can **change the behavior or curve** of our hypothesis function by making it a quadratic, cubic or square root function (or any other form):
$\theta_{0}+\theta_{1}x+\theta_{2}x^{2}$
or even better for house pricing(price is not comming down further we go!):
$\theta_{0}+\theta_{1}x+\theta_{2}x^{2}+\theta_{3}x^{3}$

**Tip:** When you make polynomial regression, the **feature scaling** become more important because of the quadratic, cubic or other functions.

For the house pricing example, we have other choices too, like:
$h_{\theta}(x)=\theta_{0}+\theta{1}(size)+\theta_{2}\sqrt{(size)}$
And there is many other choices...
## Extrapolation
In polynomial interpolation, given N points, we find the $(N-1)$st degree polynomial that we can use to predict the output for any $x$. This is called extrapolation.
# Normal Equation
Another method to solve $\theta$ analytically. Assume we had this:
$J(\theta_{0},\theta_{1},...,\theta_{m})=\frac{1}{2m}\sum_{i=1}^{m}(h_{\theta}(x^{i})-y^{i})^{2}$
According to calculus, we can do this:
$\frac{\partial}{\partial \theta_{j}}J(\theta)=...=0$ (for every $j$)
So the general formula would be this:
$$\theta=(X^{T}X)^{-1}X^{T}y$$
To calculate the X:
$$
x^{i}=
\begin{bmatrix}
x_{0}^{i} \\
x_{1}^{i} \\
x_{2}^{i} \\
... \\
x_{n}^{i} \\
\end{bmatrix}
\ \ \ \ \ \ \ \
X = 
\begin{bmatrix}
..., (x^{1})^T, ... \\
..., (x^{2})^T, ... \\
... \\
..., (x^{m})^T, ... \\
\end{bmatrix}
$$
And the $y$ is the result vector.
**E.g.**

|$x_{0}$|Size(square feet)|Number of bedrooms|Number of floors|Age of home(years)|Price($1000)|
|---|---|---|---|---|---|
|1|2104|5|1|45|460|
|1|1416|3|2|40|232|
|1|1534|3|2|30|315|
|1|852|2|1|36|178|

$$
X =
\begin{bmatrix}
1 & 2104 & 5 & 1 & 45 & 460 \\
1 & 1416 & 3 & 2 & 40 & 232 \\ 
1 & 1534 & 3 & 2 & 30 & 315 \\
1 & 852 & 2 & 1 & 36 & 178 
\end{bmatrix}
\ \ \ \ \ \ \ 
y = 
\begin{bmatrix}
460 \\
232 \\
315 \\
178
\end{bmatrix}
$$
**Tip:** There is **no need** to do feature scaling with the normal equation.

|Gradient Descent|Normal Equation|
|---|---|
|Need to choose alpha|No need to choose alpha|
|Needs many iterations|No need to iterate|
|$O(kn^2)$|$O(n^3)$, need to calculate inverse of $X^TX$|
|Works well when n is large|Slow if n is very large|

With the normal equation, computing the inversion has complexity $O(n^3)$. So if we have a very large number of features, the normal equation will be slow. In practice, when n exceeds 10,000 it might be a good time to go from a normal solution to an iterative process.

## What if $X^TX$ is non-invertible
Remember that matrices without an inverse are **"singular"** or **"degenerate"**
There are two cause for this:
- Redundant features(linearly dependent, e.g. size in square feet and size in square meters)
- Too many features(e.g. $m \leqslant n$)->Delete some features, or use regularization.

