"""SEIRD physics-informed neural network and a trainable SIRD variant.

Both classes keep the public interface of ``agents.PINN_const.EINN_PINN``
(constructor arguments, ``train_model``, ``predict``, ``params.get_params_dict``),
so either can be passed to ``PINNAgent`` as ``pinn_class`` without changing the agent.
Extra constructor arguments are optional and keyword-only in practice.

Calibration switches
--------------------
EINN_PINN is the Phase-3 network: it refits trajectories with fixed, expert-adjusted
beta, gamma and mu (``requires_grad=False``), trains everything with one Adam rate of
1e-4 and keeps the last iterate. Phase-1 calibration needs three changes. Each is an
optional argument whose default reproduces EINN_PINN:

* ``trainable_params`` - learn the epidemic parameters.
* ``param_lr`` - give them their own Adam group. Adam moves a parameter by about its
  learning rate per step, so at 1e-4 a logit-space latent travels at most ~1 unit in
  10,000 epochs and the parameters are still drifting when training stops. The
  network keeps 1e-4.
* ``restore_best`` - after training, reload the network and the parameters from the
  epoch with the lowest total loss (evaluated before that epoch's update). Without it
  Adam's loss spikes decide which state the last epoch happens to return.

When a class runs its own training loop it appends the parameter values to
``params_history`` every ``HISTORY_EVERY`` epochs and at the last epoch, and sets
``best_epoch`` (the last epoch unless ``restore_best``).

SEIRD specifics
---------------
* The exposed compartment E has no observations. The network predicts it, and it is
  constrained only by the ODE residual and by the initial condition E(0) = E0_assumed.
* The data's S column is N - I - R - D: everyone who is not yet infectious. In SEIRD
  terms that is S + E, so with ``observe_s_plus_e=True`` (default) the S channel of
  the data loss, of the initial condition and of ``predict`` is S + E, and the
  population S + E + I + R + D stays equal to N. With ``False`` the model's S alone
  is matched to the data; then the ODE conserves S + E + I + R + D = N + E0 while the
  data force S + I + R + D = N, which pins E(t) near E0 for the whole horizon.
* ``E_scaler`` (output normalization of E) is fitted on the infected training data:
  E has no data of its own, and during growth E and I are of the same order.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.PINN_const import EINN_PINN, TorchStandardScaler

HISTORY_EVERY = 100  # epochs between entries of params_history


def _adam(model):
    """EINN_PINN's optimizer, or two Adam groups when ``model.param_lr`` is set."""
    if model.param_lr is None:
        return optim.Adam(model.all_params, lr=1e-4)
    return optim.Adam([
        {"params": list(model.state_net.parameters()), "lr": 1e-4},
        {"params": list(model.params.parameters()), "lr": model.param_lr},
    ])


def _fit(model, n_epoch, lambdas, loss_terms, params_line):
    """Training loop of EINN_PINN.train_model plus the calibration switches.

    loss_terms(t_batch) returns (loss_data, loss_ode, loss_ic, loss_bc);
    params_line(values) formats the parameter line of the progress printout.
    """
    lambda_data, lambda_ode, lambda_ic, lambda_bc = lambdas
    optimizer = _adam(model)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1000, factor=0.5)
    t_batch = model.t.reshape(-1, 1).detach().requires_grad_(True)
    best_loss, best_state = float("inf"), None
    model.best_epoch = n_epoch - 1

    for epoch in range(n_epoch):
        if epoch % HISTORY_EVERY == 0 or epoch == n_epoch - 1:
            model.params_history.append({"epoch": epoch, **model.params.get_params_dict()})
        optimizer.zero_grad()
        loss_data, loss_ode, loss_ic, loss_bc = loss_terms(t_batch)
        loss = lambda_data * loss_data + lambda_ode * loss_ode + lambda_ic * loss_ic + lambda_bc * loss_bc
        value = loss.item()
        if model.restore_best and value < best_loss:  # the state this loss belongs to, before the update
            best_loss, model.best_epoch = value, epoch
            best_state = {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}
        loss.backward()
        optimizer.step()
        scheduler.step(value)

        model.losses.append(value)
        model.losses_data.append(loss_data.item())
        model.losses_ode.append(loss_ode.item())
        model.losses_ic.append(loss_ic.item())
        model.losses_bc.append(loss_bc.item())

        if epoch % 1000 == 0:
            print(f"Epoch {epoch:5d} | Loss: {value:.6f} | "
                  f"Data: {loss_data.item():.6f} | ODE: {loss_ode.item():.6f}")
            print(params_line(model.params.get_params_dict()))
            print("---")

    if best_state is not None:
        model.load_state_dict(best_state)


def _to_logit(x):
    """Inverse of ``(tanh(z) + 1) / 2``, clipped away from 0 and 1 as in EpiParams."""
    x = float(np.clip(x, 1e-12, 1 - 1e-12))
    return float(np.arctanh(2 * x - 1))


def _squash(latent):
    """Map a latent value to [0, 1]."""
    return (torch.tanh(latent) + 1) * 0.5


class EpiParamsSEIRD(nn.Module):
    """beta, sigma, gamma, mu stored in logit space and mapped to [0, 1].

    Mirrors ``EpiParams``: latents are named ``<name>_latent`` (PINNAgent overwrites
    beta, gamma and mu through these attributes) and ``get_params_dict`` returns plain
    floats. ``init_params`` may omit any parameter; missing ones take ``DEFAULTS``.
    """

    NAMES = ("beta", "sigma", "gamma", "mu")
    # beta, gamma, mu as in EpiParams; sigma = 1 / 5.2 d^-1 (mean incubation period 5.2 days)
    DEFAULTS = {"beta": 0.3, "sigma": 1 / 5.2, "gamma": 0.1, "mu": 0.04}

    def __init__(self, population_n, init_params=None, device="cpu", trainable=False):
        super().__init__()
        self.N = population_n
        self.device = device
        values = {**self.DEFAULTS, **(init_params or {})}
        unknown = sorted(set(values) - set(self.NAMES))
        if unknown:
            raise ValueError(f"unknown SEIRD parameters: {unknown}")
        for name in self.NAMES:
            latent = torch.tensor(_to_logit(values[name]), dtype=torch.float32, device=device)
            setattr(self, f"{name}_latent", nn.Parameter(latent, requires_grad=trainable))

    @property
    def beta(self):
        """Transmission rate, in [0, 1]."""
        return _squash(self.beta_latent)

    @property
    def sigma(self):
        """Rate E -> I (1 / incubation period), in [0, 1]."""
        return _squash(self.sigma_latent)

    @property
    def gamma(self):
        """Recovery rate, in [0, 1]."""
        return _squash(self.gamma_latent)

    @property
    def mu(self):
        """Mortality rate, in [0, 1]."""
        return _squash(self.mu_latent)

    def get_params_dict(self):
        """Current values as plain floats, keyed by parameter name."""
        return {name: getattr(self, name).item() for name in self.NAMES}


class EINN_PINN_SEIRD(nn.Module):
    """SEIRD counterpart of EINN_PINN with the same public interface.

    Network, loss weights, optimizer and schedule follow EINN_PINN; the output layer
    has five units (S, E, I, R, D). ``predict`` returns the four observable series in
    the data layout (S channel, I, R, D) so that PINNAgent's
    ``S_pred, I_pred, R_pred, D_pred = model.predict(t)`` and its plots work unchanged;
    ``predict_seird`` returns all five compartments.

    Extra arguments
    ---------------
    E0_assumed : float, optional
        Initial exposed population. Not observed, so it is a fixed hyperparameter.
        Default ``0.5 * I_data[0]``: the quasi-stationary ratio E/I = beta/(sigma + r)
        for beta ~ 0.1, sigma = 1/5.2 and slow growth r.
    trainable_params : bool
        Learn beta, sigma, gamma, mu (calibration) or keep them fixed (as EINN_PINN).
    observe_s_plus_e : bool
        Match S + E (True) or S alone (False) to the data's S column; see module docstring.
    param_lr : float, optional
        Separate Adam learning rate for the epidemic parameters; None keeps one group at 1e-4.
    restore_best : bool
        Return the lowest-loss state instead of the last iterate; see module docstring.
    """

    def __init__(self, t, S_data, I_data, R_data, D_data, population, train_size,
                 init_params=None, device="cpu", E0_assumed=None, trainable_params=False,
                 observe_s_plus_e=True, param_lr=None, restore_best=False):
        super().__init__()
        self.device = device
        self.N = population
        self.train_size = train_size
        self.observe_s_plus_e = observe_s_plus_e
        self.param_lr = param_lr
        self.restore_best = restore_best
        self.params_history = []
        self.best_epoch = None

        self.t = torch.tensor(t, dtype=torch.float, device=device, requires_grad=True)
        self.S_data = torch.tensor(S_data, dtype=torch.float, device=device)
        self.I_data = torch.tensor(I_data, dtype=torch.float, device=device)
        self.R_data = torch.tensor(R_data, dtype=torch.float, device=device)
        self.D_data = torch.tensor(D_data, dtype=torch.float, device=device)
        self.E0_assumed = 0.5 * float(I_data[0]) if E0_assumed is None else float(E0_assumed)

        # Output normalization, fitted on the training slice only (as in EINN_PINN).
        self.S_scaler = TorchStandardScaler()
        self.E_scaler = TorchStandardScaler()
        self.I_scaler = TorchStandardScaler()
        self.R_scaler = TorchStandardScaler()
        self.D_scaler = TorchStandardScaler()
        self.S_scaler.fit(self.S_data[:train_size].cpu().numpy(), device)
        self.E_scaler.fit(self.I_data[:train_size].cpu().numpy(), device)  # E has no data
        self.I_scaler.fit(self.I_data[:train_size].cpu().numpy(), device)
        self.R_scaler.fit(self.R_data[:train_size].cpu().numpy(), device)
        self.D_scaler.fit(self.D_data[:train_size].cpu().numpy(), device)

        self.state_net = nn.Sequential(
            nn.Linear(1, 128),
            nn.Tanh(),
            nn.Dropout(0.0),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Dropout(0.0),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Dropout(0.0),
            nn.Linear(128, 5),  # S, E, I, R, D
        ).to(device)

        self.params = EpiParamsSEIRD(self.N, init_params=init_params, device=device, trainable=trainable_params)
        self.all_params = list(self.state_net.parameters()) + list(self.params.parameters())

        self.losses = []
        self.losses_data = []
        self.losses_ode = []
        self.losses_ic = []
        self.losses_bc = []

    # ------------------------------------------------------------------ forward pass

    @staticmethod
    def _inverse(scaler, column):
        """Undo standardization for one output column (out of place)."""
        return (column.reshape(-1, 1) * scaler.std + scaler.mean).reshape(-1)

    def denormalize_states(self, states_norm):
        """Network output (n, 5) -> non-negative S, E, I, R, D in real units."""
        scalers = (self.S_scaler, self.E_scaler, self.I_scaler, self.R_scaler, self.D_scaler)
        return tuple(F.softplus(self._inverse(scaler, states_norm[:, j])) for j, scaler in enumerate(scalers))

    def _observe(self, S, E):
        """The model quantity compared with the data's S column."""
        return S + E if self.observe_s_plus_e else S

    def forward_seird(self, t_batch):
        """All five compartments S, E, I, R, D at the given times."""
        return self.denormalize_states(self.state_net(t_batch.reshape(-1, 1)))

    def forward(self, t_batch):
        """Observable compartments (S channel, I, R, D), like EINN_PINN.forward."""
        S, E, I, R, D = self.forward_seird(t_batch)
        return self._observe(S, E), I, R, D

    def compute_ode_residual(self, t_batch):
        """SEIRD residuals and states.

        Returns ``(f_S, f_E, f_I, f_R, f_D), (S, E, I, R, D)`` where each f is
        d(compartment)/dt minus the right-hand side of

            dS/dt = -beta S I / N
            dE/dt =  beta S I / N - sigma E
            dI/dt =  sigma E - gamma I - mu I
            dR/dt =  gamma I
            dD/dt =  mu I
        """
        S, E, I, R, D = self.forward_seird(t_batch)
        S_t, E_t, I_t, R_t, D_t = (
            torch.autograd.grad(x.sum(), t_batch, create_graph=True)[0].reshape(-1) for x in (S, E, I, R, D)
        )
        beta, sigma, gamma, mu = self.params.beta, self.params.sigma, self.params.gamma, self.params.mu
        infection = beta * S * I / self.N
        f_S = S_t + infection
        f_E = E_t - (infection - sigma * E)
        f_I = I_t - (sigma * E - gamma * I - mu * I)
        f_R = R_t - gamma * I
        f_D = D_t - mu * I
        return (f_S, f_E, f_I, f_R, f_D), (S, E, I, R, D)

    # ------------------------------------------------------------------ training

    def _loss_terms(self, t_batch):
        """loss_data, loss_ode, loss_ic, loss_bc for the current state (see train_model)."""
        (f_S, f_E, f_I, f_R, f_D), (S, E, I, R, D) = self.compute_ode_residual(t_batch)
        S_obs = self._observe(S, E)
        n = self.train_size
        loss_data = (F.mse_loss(S_obs[:n], self.S_data[:n])
                     + 10.0 * F.mse_loss(I[:n], self.I_data[:n])
                     + F.mse_loss(R[:n], self.R_data[:n])
                     + 1.0 * F.mse_loss(D[:n], self.D_data[:n]))
        loss_ode = torch.mean(f_S ** 2 + f_E ** 2 + f_I ** 2 + f_R ** 2 + f_D ** 2)
        loss_ic = ((S_obs[0] - self.S_data[0]) ** 2
                   + (E[0] - self.E0_assumed) ** 2
                   + (I[0] - self.I_data[0]) ** 2
                   + (R[0] - self.R_data[0]) ** 2
                   + (D[0] - self.D_data[0]) ** 2)
        loss_bc = I[-1] ** 2
        return loss_data, loss_ode, loss_ic, loss_bc

    def train_model(self, n_epoch=20000, lambda_data=1.0, lambda_ode=0.1, lambda_ic=0.1, lambda_bc=0.1):
        """Same loss structure, optimizer and schedule as EINN_PINN.train_model.

        loss_data: MSE on the training slice, weights S:I:R:D = 1:10:1:1 (S channel per
                   ``observe_s_plus_e``); E has no data term.
        loss_ode : mean of the five squared residuals.
        loss_ic  : squared deviations at t = 0 of the S channel, I, R, D from the data and
                   of E from ``E0_assumed``.
        loss_bc  : I(t_end)^2.

        ``param_lr`` and ``restore_best`` act as described in the module docstring.
        """
        _fit(self, n_epoch, (lambda_data, lambda_ode, lambda_ic, lambda_bc), self._loss_terms,
             lambda p: f"Params: β={p['beta']:.4f}, σ={p['sigma']:.4f}, γ={p['gamma']:.4f}, μ={p['mu']:.4f}")

    # ------------------------------------------------------------------ prediction

    def _time_batch(self, t_values):
        if t_values is None:
            t_values = self.t.detach().cpu().numpy()
        return torch.tensor(t_values, dtype=torch.float, device=self.device).reshape(-1, 1)

    def predict(self, t_values=None):
        """Observable series (S channel, I, R, D) on CPU, like EINN_PINN.predict."""
        with torch.no_grad():
            S_obs, I, R, D = self.forward(self._time_batch(t_values))
        return S_obs.cpu(), I.cpu(), R.cpu(), D.cpu()

    def predict_seird(self, t_values=None):
        """All five compartments S, E, I, R, D on CPU."""
        with torch.no_grad():
            states = self.forward_seird(self._time_batch(t_values))
        return tuple(x.cpu() for x in states)

    def predict_with_uncertainty(self, t_values=None, n_passes=100, dropout_rate=0.05):
        """MC Dropout, as EINN_PINN.predict_with_uncertainty; adds an 'E' entry.

        'S' is the S channel (see ``observe_s_plus_e``); 'E' is the latent compartment.
        Dropout probabilities, network mode and parameters are restored afterwards.
        """
        if t_values is None:
            t_values = self.t.detach().cpu().numpy()
        t_batch = self._time_batch(t_values)
        saved = {name: getattr(self.params, f"{name}_latent").detach().clone() for name in EpiParamsSEIRD.NAMES}
        dropout_layers = [m for m in self.state_net.modules() if isinstance(m, nn.Dropout)]
        for layer in dropout_layers:
            layer.p = dropout_rate
        was_training = self.state_net.training
        self.state_net.train()

        samples = {key: [] for key in ("S", "E", "I", "R", "D")}
        with torch.no_grad():
            for _ in range(n_passes):
                S, E, I, R, D = self.forward_seird(t_batch)
                for key, value in zip(("S", "E", "I", "R", "D"), (self._observe(S, E), E, I, R, D)):
                    samples[key].append(value.cpu().numpy())

        self.state_net.train(was_training)
        for layer in dropout_layers:
            layer.p = 0.0
        for name, value in saved.items():
            getattr(self.params, f"{name}_latent").data = value

        arrays = {key: np.array(value) for key, value in samples.items()}
        return {
            "mean": {k: v.mean(axis=0) for k, v in arrays.items()},
            "std": {k: v.std(axis=0) for k, v in arrays.items()},
            "ci_lower_95": {k: np.percentile(v, 2.5, axis=0) for k, v in arrays.items()},
            "ci_upper_95": {k: np.percentile(v, 97.5, axis=0) for k, v in arrays.items()},
            "n_passes": n_passes,
            "t": t_values,
        }


class EINN_PINN_SIRD(EINN_PINN):
    """EINN_PINN with the calibration switches of EINN_PINN_SEIRD.

    Network and losses are EINN_PINN's own. With the defaults (frozen parameters, no
    ``param_lr``, no ``restore_best``) it is identical to EINN_PINN, and ``train_model``
    is EINN_PINN's. With ``param_lr`` or ``restore_best`` set, ``train_model`` runs the
    same losses in the shared loop.
    """

    def __init__(self, t, S_data, I_data, R_data, D_data, population, train_size,
                 init_params=None, device="cpu", trainable_params=False, param_lr=None, restore_best=False):
        super().__init__(t, S_data, I_data, R_data, D_data, population, train_size,
                         init_params=init_params, device=device)
        for latent in self.params.parameters():
            latent.requires_grad_(trainable_params)
        self.param_lr = param_lr
        self.restore_best = restore_best
        self.params_history = []
        self.best_epoch = None

    def _loss_terms(self, t_batch):
        """The four loss terms of EINN_PINN.train_model, computed the same way."""
        f_S, f_I, f_R, f_D, S, I, R, D = self.compute_ode_residual(t_batch)
        n = self.train_size
        loss_data = (F.mse_loss(S[:n], self.S_data[:n]) + 10.0 * F.mse_loss(I[:n], self.I_data[:n])
                     + F.mse_loss(R[:n], self.R_data[:n]) + 1.0 * F.mse_loss(D[:n], self.D_data[:n]))
        loss_ode = torch.mean(f_S ** 2 + f_I ** 2 + f_R ** 2 + f_D ** 2)
        loss_ic = torch.mean((S[0] - self.S_data[0]) ** 2 + (I[0] - self.I_data[0]) ** 2
                             + (R[0] - self.R_data[0]) ** 2 + (D[0] - self.D_data[0]) ** 2)
        loss_bc = torch.mean(I[-1] ** 2)
        return loss_data, loss_ode, loss_ic, loss_bc

    def train_model(self, n_epoch=20000, lambda_data=1.0, lambda_ode=0.1, lambda_ic=0.1, lambda_bc=0.1):
        """EINN_PINN.train_model, or the shared loop when a calibration switch is set."""
        if self.param_lr is None and not self.restore_best:
            super().train_model(n_epoch=n_epoch, lambda_data=lambda_data, lambda_ode=lambda_ode,
                                lambda_ic=lambda_ic, lambda_bc=lambda_bc)
            self.best_epoch = len(self.losses) - 1
            return
        _fit(self, n_epoch, (lambda_data, lambda_ode, lambda_ic, lambda_bc), self._loss_terms,
             lambda p: f"Params: β={p['beta']:.4f}, γ={p['gamma']:.4f}, μ={p['mu']:.4f}")
