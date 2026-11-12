import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class EpiParams(nn.Module):
    '''Пока только SIRD'''

    def __init__(self, population_n, init_params=None, device='cpu'):
        super().__init__()
        self.N = population_n
        self.device = device

        # значения по
        ''' TODO понять бы откуда брать beta, mu 
        и как их начальные значения влияют на последующий результат'''
        if init_params is None:
            init_params = {'beta': 0.3, 'gamma': 0.1, 'mu': 0.04}

        # Инициализация в логитовом пространстве (как в статье)
        def to_logit(x):
            # Случаи на границах, чтобы не уйти в бесконечность
            if x <= 0:
                x_adj = 1e-12
            elif x >= 1:
                x_adj = 1 - 1e-12
            else:
                x_adj = x

            return np.arctanh(2 * x_adj - 1)

        self.beta_latent = nn.Parameter(torch.tensor(
            to_logit(init_params['beta'])), requires_grad=False)
        self.gamma_latent = nn.Parameter(torch.tensor(
            to_logit(init_params['gamma'])), requires_grad=False)
        self.mu_latent = nn.Parameter(torch.tensor(
            to_logit(init_params['mu'])), requires_grad=False)

    @property
    def beta(self):
        """β ∈ [0,1]"""
        return 1.0*(torch.tanh(self.beta_latent) + 1) * 0.5 - 0.0

    @property
    def gamma(self):
        """γ ∈ [0,1]"""
        return (torch.tanh(self.gamma_latent) + 1) * 0.5 + 0.00

    @property
    def mu(self):
        """μ ∈ [0,1]"""
        return (torch.tanh(self.mu_latent) + 1) * 0.5

    def get_params_dict(self):
        """Для логирования"""
        return {
            'beta': self.beta.item(),
            'gamma': self.gamma.item(),
            'mu': self.mu.item()
        }


class TorchStandardScaler:
    def fit(self, x, device):
        x = torch.tensor(x).float().to(device)
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, unbiased=False, keepdim=True)

    def transform(self, x):
        if torch.is_tensor(x):
            x -= self.mean
            x /= (self.std + 1e-7)
        else:
            x -= self.mean.cpu().numpy()
            x /= (self.std + 1e-7).cpu().numpy()
        return x

    def fit_transform(self, x, device):
        self.fit(x, device)
        return self.transform(x)

    def inverse_transform(self, x):
        x *= self.std
        x += self.mean
        return x

class EINN_PINN(nn.Module):
    '''SIRD'''

    def __init__(self, t, S_data, I_data, R_data, D_data, population, train_size, init_params=None, device='cpu'):
        super().__init__()
        self.device = device
        self.N = population
        self.train_size = train_size
        # TODO инициализация весов дефолтная корректная? МБ xavier_uniform_ сделать?

        # Данные (оставляем как есть, в реальных единицах)
        self.t = torch.tensor(t, dtype=torch.float,
                              device=device, requires_grad=True)
        self.S_data = torch.tensor(S_data, dtype=torch.float, device=device)
        self.I_data = torch.tensor(I_data, dtype=torch.float, device=device)
        self.R_data = torch.tensor(R_data, dtype=torch.float, device=device)
        self.D_data = torch.tensor(D_data, dtype=torch.float, device=device)

        # СОЗДАЕМ И ОБУЧАЕМ НОРМАЛАЙЗЕРЫ
        self.S_scaler = TorchStandardScaler()
        self.I_scaler = TorchStandardScaler()
        self.R_scaler = TorchStandardScaler()
        self.D_scaler = TorchStandardScaler()

        # Обучаем на тренировочных данных
        self.S_scaler.fit(self.S_data[:train_size].cpu().numpy(), device)
        self.I_scaler.fit(self.I_data[:train_size].cpu().numpy(), device)
        self.R_scaler.fit(self.R_data[:train_size].cpu().numpy(), device)
        self.D_scaler.fit(self.D_data[:train_size].cpu().numpy(), device)

        # Нейросеть для состояний (предсказывает в нормализованном пространстве)
        # TODO почему именно такая сеть?
        # степень двойки, чтобы лучше считать на CPU\GPU?
        # self.state_net = nn.Sequential(
        #     nn.Linear(1, 128),
        #     nn.Tanh(),
        #     nn.Linear(128, 256),
        #     nn.Tanh(),
        #     nn.Linear(256, 128),
        #     nn.Tanh(),
        #     nn.Linear(128, 4)  # S, I, R, D
        # ).to(device)
        self.state_net = nn.Sequential(
            nn.Linear(1, 128),
            nn.Tanh(),
            nn.Dropout(0.0),  # по умолчанию выключен
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Dropout(0.0),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Dropout(0.0),
            nn.Linear(128, 4)
        ).to(device)

        self.params = EpiParams(self.N,init_params = init_params, device=device)

        # Параметры для оптимизации
        self.all_params = list(self.state_net.parameters()) + \
            list(self.params.parameters())

        self.losses = []
        self.losses_data = []
        self.losses_ode = []
        self.losses_ic = []
        self.losses_bc = []

        # В __init__ вызвать:
    #     self._init_weights()

    # def _init_weights(self):
    #         for m in self.state_net.modules():
    #             if isinstance(m, nn.Linear):
    #                 nn.init.xavier_uniform_(m.weight)
    #                 nn.init.zeros_(m.bias)

    def denormalize_states(self, states_norm):
        '''Из нормализованного в реальный масштаб'''
        S_norm, I_norm, R_norm, D_norm = states_norm[:,
                                                     0], states_norm[:, 1],  states_norm[:, 2],  states_norm[:, 3]

        S = self.S_scaler.inverse_transform(S_norm.reshape(-1, 1)).reshape(-1)
        I = self.I_scaler.inverse_transform(I_norm.reshape(-1, 1)).reshape(-1)
        R = self.R_scaler.inverse_transform(R_norm.reshape(-1, 1)).reshape(-1)
        D = self.D_scaler.inverse_transform(D_norm.reshape(-1, 1)).reshape(-1)

        # Гарантируется неотрицательность (физический смысл)
        # [0, +oo]
        S = F.softplus(S)
        I = F.softplus(I)
        R = F.softplus(R)
        D = F.softplus(D)
        # Было чисто для проверки, что softplus не влияет на рассогласование
        # S = torch.clamp(S, min=0.0)
        # I = torch.clamp(I, min=0.0)
        # R = torch.clamp(R, min=0.0)
        # D = torch.clamp(D, min=0.0)

        return S, I, R, D

    # def denormalize_states(self, states_norm):
    #     S, I, R, D = states_norm[:, 0], states_norm[:, 1], states_norm[:, 2], states_norm[:, 3]
    #     S = torch.clamp(S, min=0.0)
    #     I = torch.clamp(I, min=0.0)
    #     R = torch.clamp(R, min=0.0)
    #     D = torch.clamp(D, min=0.0)
    #     return S, I, R, D

    def forward(self, t_batch):
        '''Прямой проход'''
        t_flat = t_batch.reshape(-1, 1)
        states_norm = self.state_net(t_flat)
        return self.denormalize_states(states_norm)

    def compute_ode_residual(self, t_batch):
        '''Невязка по уравнениям SIRD в реальных единицах'''
        S, I, R, D = self.forward(t_batch)

        # Производные через autograd
        S_t = torch.autograd.grad(S.sum(), t_batch, create_graph=True)[
            0].squeeze()
        I_t = torch.autograd.grad(I.sum(), t_batch, create_graph=True)[
            0].squeeze()
        R_t = torch.autograd.grad(R.sum(), t_batch, create_graph=True)[
            0].squeeze()
        D_t = torch.autograd.grad(D.sum(), t_batch, create_graph=True)[
            0].squeeze()

        # Параметры эпидемии в реальных единицах
        beta, gamma, mu = self.params.beta, self.params.gamma, self.params.mu

        # Уравнения SIRD
        dS = - beta * I * S / self.N
        dI = beta * I * S / self.N - gamma * I - mu * I
        dR = gamma * I
        dD = mu * I

        # Расчет невязки
        f_S = S_t - dS
        f_I = I_t - dI
        f_R = R_t - dR
        f_D = D_t - dD

        return f_S, f_I, f_R, f_D, S, I, R, D

    def train_model(self, n_epoch=20000, lambda_data=1.0, lambda_ode=0.1, lambda_ic = 0.1, lambda_bc = 0.1):
        '''Обучение модели'''
        optimizer = optim.Adam(self.all_params, lr=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=1000, factor=0.5)
        t_batch = self.t.reshape(-1, 1).detach().requires_grad_(True)

        for epoch in range(n_epoch):
            optimizer.zero_grad()

            # Невязки ODE
            f_S, f_I, f_R, f_D, S, I, R, D = self.compute_ode_residual(t_batch)

            # Ошибка по данным (только на обучающей выборке)
            loss_data_S = F.mse_loss(
                S[:self.train_size], self.S_data[:self.train_size])
            loss_data_I = F.mse_loss(
                I[:self.train_size], self.I_data[:self.train_size])
            loss_data_R = F.mse_loss(
                R[:self.train_size], self.R_data[:self.train_size])
            loss_data_D = F.mse_loss(
                D[:self.train_size], self.D_data[:self.train_size])

            loss_data = loss_data_S + 10.0 * loss_data_I + loss_data_R + 1.0 * loss_data_D

            # Невязка по уравнениям
            loss_ode = torch.mean(f_S**2 + f_I**2 + f_R**2 + f_D**2)

            loss_ic = torch.mean((S[0]-self.S_data[0])**2 +
                                 (I[0]-self.I_data[0])**2 + 
                                 (R[0]-self.R_data[0])**2 +
                                 (D[0]-self.D_data[0])**2)

            loss_bc = torch.mean((I[-1])**2)

            # Общая потеря
            loss = lambda_data * loss_data + lambda_ode * loss_ode + lambda_ic * loss_ic + lambda_bc * loss_bc 

            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            self.losses.append(loss.item())
            self.losses_data.append(loss_data.item())
            self.losses_ode.append(loss_ode.item())
            self.losses_ic.append(loss_ic.item())
            self.losses_bc.append(loss_bc.item())

            if epoch % 1000 == 0:
                params = self.params.get_params_dict()
                print(f'Epoch {epoch:5d} | Loss: {loss.item():.6f} | '
                      f'Data: {loss_data.item():.6f} | ODE: {loss_ode.item():.6f}')
                print(
                    f'Params: β={params["beta"]:.4f}, γ={params["gamma"]:.4f}, μ={params["mu"]:.4f}')
                print('---')

    def predict(self, t_values=None):
        '''Получение предсказаний'''
        if t_values is None:
            t_values = self.t.detach().cpu().numpy()

        t_batch = torch.tensor(t_values, dtype=torch.float,
                               device=self.device).reshape(-1, 1)

        with torch.no_grad():
            S, I, R, D = self.forward(t_batch)

        return S.cpu(), I.cpu(), R.cpu(), D.cpu()
    
    def predict_with_uncertainty(self, t_values=None, n_passes=100, dropout_rate=0.05):
        """
        MC Dropout prediction with temporarily enabled dropout.
        Returns uncertainty for ALL compartments (S, I, R, D).
        """
        if t_values is None:
            t_values = self.t.detach().cpu().numpy()
        
        t_batch = torch.tensor(t_values, dtype=torch.float, 
                            device=self.device).reshape(-1, 1)
        
        # Сохраняем текущие параметры
        original_beta = self.params.beta_latent.clone()
        original_gamma = self.params.gamma_latent.clone()
        original_mu = self.params.mu_latent.clone()
        
        # Находим все Dropout слои и временно включаем их
        dropout_layers = []
        for module in self.state_net.modules():
            if isinstance(module, nn.Dropout):
                dropout_layers.append(module)
                module.p = dropout_rate
        
        # Включаем режим training
        was_training = self.state_net.training
        self.state_net.train()
        
        all_S = []
        all_I = []
        all_R = []
        all_D = []
        
        with torch.no_grad():
            for _ in range(n_passes):
                t_flat = t_batch.reshape(-1, 1)
                states_norm = self.state_net(t_flat)
                S, I, R, D = self.denormalize_states(states_norm)
                all_S.append(S.cpu().numpy())
                all_I.append(I.cpu().numpy())
                all_R.append(R.cpu().numpy())
                all_D.append(D.cpu().numpy())
        
        # Восстанавливаем
        self.state_net.train(was_training)
        for module in dropout_layers:
            module.p = 0.0
        
        # Восстанавливаем параметры
        self.params.beta_latent.data = original_beta
        self.params.gamma_latent.data = original_gamma
        self.params.mu_latent.data = original_mu
        
        all_S = np.array(all_S)
        all_I = np.array(all_I)
        all_R = np.array(all_R)
        all_D = np.array(all_D)
        
        return {
            'mean': {
                'S': np.mean(all_S, axis=0),
                'I': np.mean(all_I, axis=0),
                'R': np.mean(all_R, axis=0),
                'D': np.mean(all_D, axis=0)
            },
            'std': {
                'S': np.std(all_S, axis=0),
                'I': np.std(all_I, axis=0),
                'R': np.std(all_R, axis=0),
                'D': np.std(all_D, axis=0)
            },
            'ci_lower_95': {
                'S': np.percentile(all_S, 2.5, axis=0),
                'I': np.percentile(all_I, 2.5, axis=0),
                'R': np.percentile(all_R, 2.5, axis=0),
                'D': np.percentile(all_D, 2.5, axis=0)
            },
            'ci_upper_95': {
                'S': np.percentile(all_S, 97.5, axis=0),
                'I': np.percentile(all_I, 97.5, axis=0),
                'R': np.percentile(all_R, 97.5, axis=0),
                'D': np.percentile(all_D, 97.5, axis=0)
            },
            'n_passes': n_passes,
            't': t_values
        }