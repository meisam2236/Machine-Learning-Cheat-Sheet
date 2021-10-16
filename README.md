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


# Cost Function
We can measure the accuracy of our hypothesis function by using a cost function. The goal is to minimize the cost function.
## Loss Function
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
# Linear Algebra
## Matrix
Rectangular array of numbers.
$$
\begin{equation*}
A = 
\begin{bmatrix}
1402 & 191 \\
1371 & 821 \\
949 & 1437 \\
147 & 1448
\end{bmatrix}
\end{equation*}
$$
### Dimensions of matrix
Number of rows x number of columns
In the example above, dim is 4x2
### Matrix Elements (entries of matrix)
$A_{ij}=$ "$i,j$ entry" in the $i^{th}$ row, $j^{th}$ column.
$A_{1 1}=1402$
$A_{3 2}=1437$
$A_{1 2}=191$
$A_{4 3}=undefiend$
## Vector
An nx1 matrix.
$$
\begin{equation*}
y = 
\begin{bmatrix}
460 \\
232 \\
315 \\
178
\end{bmatrix}
\end{equation*}
$$
The example above is a 4-dimensional vector.
$y_{i}=$ $i^{th}$ element
$y_{1}=460$
$y_{2}=232$
$y_{3}=315$
We've also got 0-indexed vector!
**Tip:** Usually we refer uppercase for matrices and lowercase for vectors.
## Matrix Addition
$$
\begin{bmatrix}
1 & 0 \\
2 & 5 \\
3 & 1
\end{bmatrix}
+
\begin{bmatrix}
4 & 0.5 \\
2 & 5 \\
0 & 1
\end{bmatrix}
=
\begin{bmatrix}
5 & 0.5 \\
4 & 10 \\
3 & 2
\end{bmatrix}
$$
**Tip:** We can not add matrices with different dimensions.
## Scalar(real number) Multiplication
$$
3\times
\begin{bmatrix}
1 & 0 \\
2 & 5 \\
3 & 1
\end{bmatrix}
=
\begin{bmatrix}
3 & 0 \\
6 & 15 \\
9 & 3
\end{bmatrix}
$$

$$
\begin{bmatrix}
4 & 0 \\
6 & 3
\end{bmatrix}
/4=
\frac{1}{4}\times
\begin{bmatrix}
4 & 0 \\
6 & 3
\end{bmatrix}
=
\begin{bmatrix}
1 & 0 \\
\frac{3}{2} & \frac{3}{4}
\end{bmatrix}
$$
## Combination of Operands
$$
3\times
\begin{bmatrix}
1 \\
4 \\
2
\end{bmatrix}
+
\begin{bmatrix}
0 \\
0 \\
5
\end{bmatrix}
-
\begin{bmatrix}
3 \\
0 \\
2
\end{bmatrix}
/3=
\begin{bmatrix}
3 \\
12 \\
6
\end{bmatrix}
+
\begin{bmatrix}
0 \\
0 \\
5
\end{bmatrix}
-
\begin{bmatrix}
1 \\
0 \\
\frac{2}{3}
\end{bmatrix}
=
\begin{bmatrix}
2 \\
12 \\
10\frac{1}{3}
\end{bmatrix}
$$
## Matrix Vector Multiplication
$$
\begin{bmatrix}
1 & 3 \\
4 & 0 \\
2 & 1
\end{bmatrix}
\begin{bmatrix}
1 \\
5
\end{bmatrix}
=
\begin{bmatrix}
(1\times1)+(3\times5) \\
(4\times1)+(0\times5) \\
(2\times1)+(1\times5)
\end{bmatrix}
=
\begin{bmatrix}
16 \\
4 \\
7
\end{bmatrix}
$$
**Explanation:** We multiply first one $i^{th}$ row with elements of second one $j^{th}$ column and add them up. So the new dimension will be (number of row of the first one) x (number of column of the second one).

## Turn hypothesis to vector matrix multiplication(More efficient!)
Assume we have house sizes and wanted to predict the price:
House sizes: 2104 - 1416 - 1534 - 852
hypothesis: $h_{\theta}(x)=-40+0.25x$
$$
\begin{bmatrix}
1 & 2104 \\
1 & 1416 \\
1 & 1534 \\
1 & 852
\end{bmatrix}
\times
\begin{bmatrix}
-40 \\
0.25
\end{bmatrix}
=
\begin{bmatrix}
(-40\times1)+(0.25\times2104) \\
(-40\times1)+(0.25\times1416) \\
(-40\times1)+(0.25\times1534) \\
(-40\times1)+(0.25\times852) \\
\end{bmatrix}
$$
**And the code will be**-> prediction = data_matrx * parameters

## Matrix Matrix Multiplication
$$
\begin{bmatrix}
1 & 3 & 2 \\
4 & 0 & 1 
\end{bmatrix}
\begin{bmatrix}
1 & 3 \\
0 & 1 \\
5 & 2
\end{bmatrix}
=
\begin{bmatrix}
(1\times1)+(3\times0)+(2\times5) & (1\times3)+(3\times1)+(2\times2) \\
(4\times1)+(0\times0)+(1\times5) & (4\times3)+(0\times1)+(1\times2)
\end{bmatrix}
=
\begin{bmatrix}
11 & 10 \\
9 & 14 
\end{bmatrix}
$$
## Turn hypothesis to matrix matrix multiplication(More efficient!)
Assume we have house sizes and wanted to predict the price:
House sizes: 2104 - 1416 - 1534 - 852
Having 3 competing hypothesis:
1. $h_{\theta}(x)=-40+0.25x$
2. $h_{\theta}(x)=200+0.1x$
3. $h_{\theta}(x)=-150+0.4x$

$$
\begin{bmatrix}
1 & 2104 \\
1 & 1416 \\
1 & 1534 \\
1 & 852
\end{bmatrix}
\times
\begin{bmatrix}
-40 & 200 & -150 \\
0.25 & 0.1 & 0.4
\end{bmatrix}
=
\begin{bmatrix}
486 & 410 & 692 \\
314 & 342 & 416 \\
344 & 353 & 464 \\
173 & 285 & 191 \\
\end{bmatrix}
$$
**Explanation:** The first column would be the prediction of first $h_{\theta}(x)$, the second column would be the prediction of second $h_{\theta}(x)$ and so on.

In matrices multiplication we have two rules:
1.  not commutative-> $A \times B\neq B \times A$
2. associative-> $A \times B \times C = A \times (B \times C) = (A \times B) \times C$

## Identity Matrix
Denoted $I$(or $I_{n \times n}$).

Example:
$$
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
$$
$$
\begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}
$$
$$
\begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

For any matrix $A$,
$A \times I = I \times A = A$
Remember that we said $A \times B\neq B \times A$, but if B is an identity matrix, $A \times B = B \times A$ !
## Matrix Inverse
If $A$ is an $m \times m$ matrix, and if it has an inverse,
$A.A^{-1}=A^{-1}.A=I$
**Note that the matrix should be a squre matrix.**
$$
\begin{bmatrix}
3 & 4 \\
2 & 16
\end{bmatrix}
\begin{bmatrix}
0.4 & -0.1 \\
-0.05 & 0.075 
\end{bmatrix}
=
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
= I_{2\times2}
$$
Matrices that don't have an inverse are **"singular"** or **"digenerate"**.like this:
$$
\begin{bmatrix}
0 & 0 \\
0 & 0
\end{bmatrix}
$$
## Matrix Transpose
Let $A$ be an $m \times n$ matrix, and let $B=A^T$.
Then $B$ is an $n \times m$ matrix and $B_{ij}=A_{ji}$
Suppose we have this matrice:
$$
A = 
\begin{bmatrix}
1 & 2 & 0 \\
3 & 5 & 9
\end{bmatrix}
$$
Transpose of this matrix should be:
$$
A^T = 
\begin{bmatrix}
1 & 3 \\
2 & 5 \\
0 & 9
\end{bmatrix}
$$
The way that we transpose a matrix is that first row will be the first column, second row will be the second column and so on.
