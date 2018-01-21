# Stochastic Descent Algorithms

The goal of this repository is to minimize a non-convex function
$$
\begin{equation}
\min_{x} F(x),
\end{equation}
$$
In order to attain a global minimum we add a noise to three different descent algorithms

Gradient Descent:
$$
\begin{equation}
X^{n+1} = X^n - \frac{1}{\gamma} \nabla_X F(X^n) \Delta t_n + \frac{\sigma_n}{\gamma} \Delta W^n\
\end{equation}
$$
Moment 1 (Langevin):
$$
\begin{equation}
\left\{
\begin{aligned}
V^{n+1} &=  (1-\mu \Delta t_n)V^n - \frac{1}{\gamma} \nabla_X F(X^n) \Delta t_n + \frac{\sigma_n}{\gamma} \Delta W^n\\
X^{n+1} &= X^n + V^{n+1}\Delta t_n \\
\end{aligned}
\right.
\end{equation}
$$
Moment 2:
$$
\begin{equation} \label{equ:moment2}
\left\{
\begin{aligned}
z^{n+1} & = -\lambda_1 z^n\Delta t +\lambda_2 V^n d t + \frac{\sigma_n}{\gamma} \Delta W^n\\
V^{n+1} &=  (1-\mu \Delta t_n)V^n - \frac{1}{\gamma} \nabla_X F(X^n) \Delta t_n - z^{n+1}\Delta t_n.\\
X^{n+1} &= X^n + V^{n+1}\Delta t_n \\
\end{aligned}
\right.
\end{equation}
$$
where $\sigma_n \to 0$, $\Delta t_n \to 0$, $\Delta W^n \sim N(0,1)$

