# Numerical-Methods
I took a course in numerical methods. This repository is the result of me coding everything I learned in that course.

## Interpolation
Given a set of points and the corresponding function values, can we estimate the original function?

I have implemented the following methods:
* Lagrange Polynomials
* Natural Cubic Spline

## Numerical Integration
Given a set of points and the corresponding function values, can we estimate the integral of the original function?

I have implemented the following methods:
* Lagrange Interpolation + Integration
* Compound Quadrature
  * Newton-Cotes (equally spaced subintervals)
    * Simpson's Rule (m=3)
    * Trapezoidal Rule (m=2)
* Guassian Quadrature

## Solving Linear System of Equations
Given a system of linear equations, find the solution.

I have implemented the following methods:
* LU Decomposition
* LUP Decomposition
* Least Squares Method

## Solving Nonlinear System of Equations
Given any system of equations, find a solution.

I have implemented the following methods:
* Bisection Method
* Secant Method
* Newton's Method
* Generalized Newton's Method (for system of vector equations)

## Relaxation Methods for Solving Systems of Equations
Given a system of equations, find a solution efficiently to a given tolerance.

I have implemented the following methods:
* Point-Jacobean
* Guass-Seidel
* Successive Over Relaxation

## Optimization
Given an objective function and set of constraints, find the minimizer.

I have implemented the following methods:
* Golden Section Search

## Taylor Table
Given a derivative estimator, find its associated error

## ODE Solvers
Given an ODE, solve it numerically using time marching.

I have implemented the following time marching methods:
* Explicit Euler
* Implicit Euler
* Leapfrog
* McCormmack
* RK4

**Note**: It is very easy to add additional methods
