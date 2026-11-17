"""SIRD PINN with trainable constant beta, gamma, mu.

The model follows the Phase-1 calibration code of the paper
(EINN_PINN in NEW_PINN/PINN_const.py): a fully connected network maps
time to standardized S, I, R, D, the outputs are rescaled and passed
through a softplus, and beta, gamma, mu are trained through latent
variables z with value (tanh z + 1) / 2.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ARCHITECTURES = {
    "128-256-128": (128, 256, 128),
}


def _to_latent(x):
    x = min(max(x, 1e-12), 1 - 1e-12)
    return np.arctanh(2 * x - 1)


class Scaler:
    """Standardization fitted on the training window (population SD)."""

    def __init__(self, x):
        x = torch.as_tensor(x, dtype=torch.float)
        self.mean = x.mean()
        self.std = x.std(unbiased=False)

    def inverse(self, z):
        return z * self.std + self.mean


class SIRDPINN(nn.Module):
    def __init__(self, t, S, I, R, D, population, train_size,
                 init_params, hidden=(128, 256, 128)):
        super().__init__()
        self.N = population
        self.train_size = train_size
        self.t = torch.tensor(t, dtype=torch.float)
        self.data = [torch.tensor(x, dtype=torch.float) for x in (S, I, R, D)]
        self.scalers = [Scaler(x[:train_size]) for x in (S, I, R, D)]

        layers, width = [], 1
        for h in hidden:
            layers += [nn.Linear(width, h), nn.Tanh()]
            width = h
        layers.append(nn.Linear(width, 4))
        self.net = nn.Sequential(*layers)

        self.z_beta = nn.Parameter(torch.tensor(_to_latent(init_params["beta"])))
        self.z_gamma = nn.Parameter(torch.tensor(_to_latent(init_params["gamma"])))
        self.z_mu = nn.Parameter(torch.tensor(_to_latent(init_params["mu"])))

    @staticmethod
    def _rate(z):
        return (torch.tanh(z) + 1) * 0.5

    def rates(self):
        return self._rate(self.z_beta), self._rate(self.z_gamma), self._rate(self.z_mu)

    def forward(self, t):
        out = self.net(t.reshape(-1, 1))
        return [F.softplus(sc.inverse(out[:, k])) for k, sc in enumerate(self.scalers)]

    def loss_terms(self, t):
        S, I, R, D = self.forward(t)
        dS, dI, dR, dD = [torch.autograd.grad(x.sum(), t, create_graph=True)[0].squeeze()
                          for x in (S, I, R, D)]
        beta, gamma, mu = self.rates()
        f_S = dS + beta * I * S / self.N
        f_I = dI - (beta * I * S / self.N - gamma * I - mu * I)
        f_R = dR - gamma * I
        f_D = dD - mu * I

        n = self.train_size
        S_obs, I_obs, R_obs, D_obs = self.data
        loss_data = (F.mse_loss(S[:n], S_obs[:n]) + 10.0 * F.mse_loss(I[:n], I_obs[:n])
                     + F.mse_loss(R[:n], R_obs[:n]) + F.mse_loss(D[:n], D_obs[:n]))
        loss_ode = torch.mean(f_S ** 2 + f_I ** 2 + f_R ** 2 + f_D ** 2)
        loss_ic = ((S[0] - S_obs[0]) ** 2 + (I[0] - I_obs[0]) ** 2
                   + (R[0] - R_obs[0]) ** 2 + (D[0] - D_obs[0]) ** 2)
        loss_bc = I[-1] ** 2
        return loss_data, loss_ode, loss_ic, loss_bc

    def fit(self, n_epoch, lambda_data=1.0, lambda_ode=1.0, lambda_ic=0.1,
            lambda_bc=0.1, lr=1e-4, log_every=0):
        params = list(self.net.parameters()) + [self.z_beta, self.z_gamma, self.z_mu]
        opt = optim.Adam(params, lr=lr)
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=1000, factor=0.5)
        t = self.t.reshape(-1, 1).clone().requires_grad_(True)
        for epoch in range(n_epoch):
            opt.zero_grad()
            l_data, l_ode, l_ic, l_bc = self.loss_terms(t)
            loss = lambda_data * l_data + lambda_ode * l_ode + lambda_ic * l_ic + lambda_bc * l_bc
            loss.backward()
            opt.step()
            sched.step(loss.item())
            if log_every and epoch % log_every == 0:
                b, g, m = (x.item() for x in self.rates())
                print(f"epoch {epoch:6d}  loss {loss.item():.4e}  data {l_data.item():.4e}  "
                      f"ode {l_ode.item():.4e}  beta {b:.4f}  gamma {g:.4f}  mu {m:.5f}",
                      flush=True)
        return loss.item(), l_data.item(), l_ode.item()

    def predict(self):
        with torch.no_grad():
            return [x.numpy() for x in self.forward(self.t)]
