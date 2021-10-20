- [Matlab/Octave](#matlaboctave)
	- [Basic Operations](#basic-operations)
	- [Moving Data Around](#moving-data-around)
	- [Computing on Data](#computing-on-data)
	- [Plotting Data](#plotting-data)
	- [Control Statements](#control-statements)
		- [for](#for)
		- [while](#while)
		- [if](#if)
- [Functions](#functions)
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
A*C % multiplication
A .* B % element-wise multiplication
A .^ 2 % element-wise square
1 ./ A % element-wise division
v = [1;2;3]
log(v) % element-wise logarithm
exp(v) % element-wise exponentiation
abs([-1;2;-3]) % element-wise absolute value
-v % same as -1 * v
v + ones(length(v), 1) % same as v + 1
A' % A-transpose
a = [1 15 2 0.5]
max(a) % maximum value of a
[val, indx] = max(a) % val will be maximum value and indx will be the index of it
max(A) % column-wise maximum value
a < 3 % element-wise comparison
find(a < 3) % give index result of the comparison
A = magic(3) % n-by-n magic function
[r, c] = find(A >= 7) % r represent row and c represent column of the comparison
sum(a) % adds up all the elements of A
prod(a) % product of a which is multiplication of all elements
floor(a) % round each element to less
ceil(a) % round each element to more
rand(n)
max(A, [], 1) % max of column
max(A, [], 2) % max of row
max(max(A)) % same as max(A(:)) which is maximum number of a matrix
A = magic(9)
sum(A, 1) % sum per-column
sum(A, 2) % sum per-row
sum(sum(A .* eye(9))) % sum of diagonal values
sum(sum(A .* flipud(eye(9)))) % sum of reverse diagonal values
A = magic(3)
temp = pinv(A) % inverse of A
temp * A % identity matrix
```
## Plotting Data
The codes below are one-line codes:
```matlab
t = [0:0.01:0.98];
y1 = sin(2*pi*4*t);
plot(t, y1); % plot with x = t and y = y1
y2 = cos(2*pi*4*t);
plot(t, y1);
hold on; % hold on the plot for the next plot
plot(t, y2, 'r'); % plot another with color 'red'
xlabel('time'); % label horizontal axis time
ylabel('value'); % label vertical axis value
legend('sin', 'cos') % legend the plot
title('my plot') % add title to plot
print -dpng 'myplot.png' % save to plot
close % close the plot
figure(1); plot(t, y1); % plot on figure 1
figure(2); plot(t, y2); % plot on figure 2
subplot(1, 2, 1); % divide the plot to 1x2 and access first element
plot(t, y1); % plot to first half
subplot(1, 2, 2); % access second element
plot(t, y2); % plot to second half
axis([0.5 1 -1 1]) % make axis for second half as 0.5 to 1 for horizontal axis and -1 to 1 for vertical axis
clf; % clear a figure
A = magic(5)
imagesc(A) % color code matrix
imagesc(A), colorbar, colormap gray; % color bar shows the shade and color map set it to grayscale
a=1, b=2, c=3 % with comma you can execute multiple command at the same time
```
## Control Statements
### for
```matlab
v = zeros(10, 1)
for i=1:10,
	v(i) = 2^i;
end;
v
```
```matlab
indicies = 1:10;
for i=indicies,
	disp(i);
end;
```
### while
```matlab
v = zeros(10, 1)
i = 1;
while i <= 5,
	v(i) = 100;
	i = i + 1;
end;
```
### if
```matlab
v = zeros(10, 1)
i = 1;
while true,
	v(i) = 999;
	i = i + 1;
	if i == 6,
		break;
	end;
end;
```
```matlab
v = zeros(10, 1)
v(1) = 2;
if v(1) == 1,
	disp('The value is one');
elseif v(1) == 2,
	disp('The value is two');
end;
```
# Functions
Creat a .m file and write your function in it:
```matlab
function y = squareThisNumber(x)
y = x^2;
```
Now use it like this:
```matlab
squareThisNumber(5)
```
**Another example:**
```matlab
function [y1, y2] = squareAndCubeThisNumber(x)

y1 = x^2;
y2 = x^3;
```
and:
```matlab
[a,b] = squareAndCubeThisNumber(5)
```
**Another one:**
Define a function to compute the cost function $J(\theta)$:
```matlab
function J = costFunction(X, y, theta)

m = size(X, 1); % number of training example
predictions = X*theta; % predicitons of hypothesis on all m examples
sqrErrors = (predictions-y).^2; % squared errors

J = 1/(2*m) * sum(sqrErrors);
```
And use it:
```matlab
X = [1 1; 1, 2; 1, 3]
y = [1; 2; 3]
theta = [0; 1]
j = costFunction(X, y, theta) % it gives 0 because theta is exactly right!
```
