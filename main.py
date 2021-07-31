from itertools import chain
from neurodiffeq import diff
from neurodiffeq.conditions import IVP
from neurodiffeq.monitors import Monitor1D
from neurodiffeq.solvers import Solver1D
from neurodiffeq.callbacks import MonitorCallback
from neurodiffeq.networks import FCNN
from neurodiffeq.generators import Generator1D
from torch.optim import LBFGS
import torch.autograd
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torch

# specify the ODE system

Lmda = 0.47596532847715467
lmbda = 0.6321
mu = 0.9704
beta1, beta2, beta3, beta4, beta5, beta6, beta7, beta8, beta9, beta10, beta11 = 0.2877, 0.7613, 0.4389, 0.1234, 0.2431, 0.4000, 0.3000, 0.2345, 0.3312, 0.4123, 0.5111
mu1, mu2, mu3, mu4, mu5, mu6 = 0.0432, 0.2006, 0.656, 0.9764, 0.6704, 0.0286


def ebola_system(S, V, E, I, R, t): return [diff(S, t) - lmbda + (mu + beta8) * S + (beta1 + beta4 + beta6) * S * E + (beta5 + beta7) * S * I,
                                            diff(V, t) - beta8 * S + (mu6 + beta11) * V + beta9 *
                                            V * E + beta10 * V * I,
                                            diff(E, t) - (beta1 + beta4 + beta6) * S * E - beta9 *
                                            V * E + beta2 * E *
                                            I + (mu1 + mu2) * E,
                                            diff(I, t) - (beta5 + beta7) * S * I - beta2 * E * I -
                                            beta10 * V * I +
                                            (beta3 + mu3 + mu4) * I,
                                            diff(R, t) - beta3 * I - beta11 * V + mu5 * R]


RO = lmbda*(beta8*beta9 + (beta1+beta4+beta6) * (mu6 + beta11)) / \
    ((mu + beta8)*(mu1 + mu2)*(mu6 + beta11))
S0 = lmbda / (mu + beta8)
V0 = (lmbda*beta8)/((mu + beta8)*(mu5 + beta11))
E0, I0 = 0, 0
R0 = (lmbda*beta8*beta11)/(mu5*(mu+beta8)*(mu5+beta11))
F0 = (round(S0, 4), round(V0, 4), round(E0, 4), round(I0, 4), round(R0, 4))
S1 = lmbda / (Lmda + mu + beta8)
A = (beta2*(mu6+beta11)+beta9*(beta3+mu3+mu4)-beta10*(mu1+mu2))*(mu1+mu2)
a = mu+beta8
B = lmbda**2*(beta1+beta4+beta6) * \
    (beta9*(beta5+beta7)-beta10*(beta1+beta4+beta6))
C = lmbda*(-beta9*beta2*beta8-beta9*(mu1+mu2)*(beta5+beta7)+(beta1+beta4+beta6)
           * (2*beta10*(mu1+mu2)-2*beta2*(mu6+beta11)-beta9*(mu6+beta11)))
c = mu+beta8
F = beta2*(beta10*(mu1+mu2)-beta2*(mu6+beta11)-beta9*(beta3+mu3+mu4))
f = mu+beta8
G = lmbda*beta2*(beta9*(beta5+beta7)-beta10*(beta1+beta4+beta6))
g = mu+beta8
X = (A * (Lmda + a)**2 + B + C * (Lmda + c))
Y = (F * (Lmda + f)**2 + G*(Lmda + g))
I1 = X/Y
V1 = (1/beta9)*(mu1+mu2) - (lmbda*(beta1+beta4+beta6)) / \
    (beta9*(Lmda+mu+beta8)) + (beta2/beta9)*I1
E1 = (1/beta2)*(mu1+mu2) - lmbda*(beta5 + beta7)/(beta2*(Lmda+mu+beta8)) - beta10*(mu1+mu2) / \
    (beta2*beta9) + lmbda*beta10*(beta1+beta4+beta6) / \
    (beta2*beta9*(Lmda+mu+beta8)) - (beta10/beta9)*I1
R1 = beta11*(mu1+mu2)/(mu5*beta9) - lmbda*beta11*(beta1+beta4+beta6) / \
    (mu5*beta9*(Lmda+mu+beta8)) + (beta3/mu5 + (beta2*beta11)/(mu5*beta9))*I1
F1 = (round(S1, 4), round(V1, 4), round(E1, 4), round(I1, 4), round(R1, 4))

# specify the initial conditions
ebola_init_vals_pc = [
    IVP(t_0=0.0, u_0=0.1),
    IVP(t_0=0.0, u_0=0.1),
    IVP(t_0=0.0, u_0=0.1),
    IVP(t_0=0.0, u_0=0.1),
    IVP(t_0=0.0, u_0=0.1)
]

monitor = Monitor1D(t_min=0, t_max=100, check_every=1)
monitor_callback = MonitorCallback(monitor)


def my_callback(solver):
    if solver.lowest_loss < 1e-6:
        solver._stop_training = True


ebola_nets_lv = [
    FCNN(n_input_units=1, n_output_units=1,
         n_hidden_units=10, actv=nn.Sigmoid),
    FCNN(n_input_units=1, n_output_units=1,
         n_hidden_units=10, actv=nn.Sigmoid),
    FCNN(n_input_units=1, n_output_units=1,
         n_hidden_units=10, actv=nn.Sigmoid),
    FCNN(n_input_units=1, n_output_units=1,
         n_hidden_units=10, actv=nn.Sigmoid),
    FCNN(n_input_units=1, n_output_units=1, n_hidden_units=10, actv=nn.Sigmoid)
]


lbfgs = LBFGS(chain.from_iterable(n.parameters()
                                  for n in ebola_nets_lv), lr=0.01, max_iter=100)


# solve the ODE system
train_gen = Generator1D(size=1024,  t_min=0.0, t_max=100, method='uniform')
valid_gen = Generator1D(size=128, t_min=0.0, t_max=100,
                        method='equally-spaced')
ebola_solver = Solver1D(ode_system=ebola_system, conditions=ebola_init_vals_pc,
                        t_min=0, t_max=100, nets=ebola_nets_lv, optimizer=lbfgs, train_generator=train_gen, valid_generator=valid_gen, n_batches_train=10, n_batches_valid=5)


ebola_solver.fit(max_epochs=2000, callbacks=[monitor_callback, my_callback])
solution_pc = ebola_solver.get_solution()

ts = np.linspace(0, 100, 10000)
s, v, e, i, r = solution_pc(ts, to_numpy=True)

fig, axes = plt.subplots(1, 5)
fig.suptitle(
    f'EVD System Solution with ANN\nR0={round(RO, 4)}\n{F0 if RO < 1 else F1}')
fig.tight_layout()
axes[0].plot(ts, s)
axes[0].set_ylabel('S')
axes[0].set_xlabel('Time')
axes[1].plot(ts, v)
axes[1].set_ylabel('V')
axes[1].set_xlabel('Time')
axes[2].plot(ts, e)
axes[2].set_ylabel('E')
axes[2].set_xlabel('Time')
axes[3].plot(ts, i)
axes[3].set_ylabel('I')
axes[3].set_xlabel('Time')
axes[4].plot(ts, r)
axes[4].set_ylabel('R')
axes[4].set_xlabel('Time')
plt.show()
