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
\ \ \ \ \ \ \
\begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}
\ \ \ \ \ \ \
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
Matrices that don't have an inverse are **"singular"** or **"degenerate"**.like this:
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
# Polynomial Regression
With polynomial, we can **change the behavior or curve** of our hypothesis function by making it a quadratic, cubic or square root function (or any other form):
$\theta_{0}+\theta_{1}x+\theta_{2}x^{2}$
or even better for house pricing(price is not comming down further we go!):
$\theta_{0}+\theta_{1}x+\theta_{2}x^{2}+\theta_{3}x^{3}$

**Tip:** When you make polynomial regression, the **feature scaling** become more important because of the quadratic, cubic or other functions.

For the house pricing example, we have other choices too, like:
$h_{\theta}(x)=\theta_{0}+\theta{1}(size)+\theta_{2}\sqrt{(size)}$
And there is many other choices...
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

# Matlab/Octave
You can first write your code in matlab or octave to make sure the project is working; Then make it with C++ or Java for better performance!
## Basic Operations
The codes below are one-line codes:
```matlab
5+6 % addition
3-2 % subtraction
5*8 % multiplication
1/2 % division
2^6 % exponent

1 == 2 % equality
1 ~= 2 % not equality
1 && 0 % AND
1 || 0 % OR
xor(1, 0) % OR

PS1('>> '); % change octave prompt

a = 3; % variable defining(semicolon supressing output)
b = 'hi'; % string assignment
c = (3>=1) % which is True(1)
a=pi % pi variable
disp(a) % more complex printing(which is display)
disp(sprintf('2 decimals: %0.2f', a)) % generate string a
format long % display in long decimal numbers
format short % display in short decimal numbers

A = [1 2; 3 4; 5 6] % define matix A
a = [1 2 3] % define row vector a
a = [1; 2; 3] % define column vector a
v = 1:6 % define a vector with values 1 to 6
v = 1:0.1:2 % define a vector with values start:step:stop
ones(2, 3) % matrix 2 by 3 with value 1
C = 2*ones(2,3) % generate matrix 2 by 3 with value 2
w = zeros(1, 3) % row vector with value 0
w = rand(1, 3) % row vector with random values
randn(1, 3) % a gaussian distribution with mean zero and standard deviation(variance) equal to one

w = -6 + sqrt(10)*(randn(1, 10000)); % complex vector
hist(w) % histogram
hist(w, 50) % histogram with 50 bins
I = eye(4)  % 4 by 4 identity matrix

help(eye) % help function for eye command

a = size(A) % size of the matrix
size(a) % a is now a vector
size(A, 1) % first dimension size(number of row)
size(A, 2) % second dimension size(number of column)
length(A) % size of a longest dimension

A(3, 2) % element of third row second column
A(2, :) % elements in second row
A([1 3], :) % all elements in first and third row
A(:,2) = [10; 11; 12]  % assign 10, 11, 12 to second column
A = [A, [100, 101, 102]]; % append another column to A
A(:) % put all elements of A into a single vector
A = [1 2; 3 4; 5 6]
B = [11 12; 13 14; 15 16]
C = [A, B] % concating A and B on the side
C = [A; B] % concating A and B on top of each other
```
## Moving Data Around
The codes below are one-line codes:
```matlab
pwd % current directory(path)
cd 'C:\users\meisa\Desktop' % change directory
ls % list files in current direcotory
load featuresX.dat % loading .dat file
load('priceY.dat') % another way of loading .dat file
who % shows all variables
whos % shows variabels with detail
clear featuresX % delete variable
v = priceY(1:10) % get first 10 elements of another variable
save hello.mat v; % save variable v to hello.mat file(binary format-compressed)
save hello.txt v -ascii % save variable v to hello.txt file(human readable)
```
## Computing on Data
The codes below are one-line codes:
```matlab
A = [1 2; 3 4; 5 6]
B = [11 12; 13 14; 15 16]
C = [1 1; 2 2]
A*C
```
