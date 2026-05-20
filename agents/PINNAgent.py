"""
PINNAgent
---------
Агент для запуска EINN_PINN с параметрами из state.
Структура аналогична SurrogateAgent.

Ожидает в state:
    generated_params      : dict  — beta, gamma, mu
    initial_conditions    : dict  — population, S0, I0, R0, D0
    simulation_params     : dict  — t_max, num_points (опционально)
    pinn_data             : dict  — S, I, R, D, train_size (реальные данные)
                            ИЛИ
    task_config.data_path : str   — путь к CSV-файлу

Записывает в state:
    pinn_results : dict с ключами success, final_params, plot_paths, losses, tag
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any, Optional

warnings.filterwarnings("ignore")


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _run_tag(iteration: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"iter{iteration}_{ts}"


class PINNAgent:
    """
    Агент для обучения EINN_PINN с параметрами, принятыми оптимизационным пайплайном.

    Parameters
    ----------
    pinn_class : type
        Класс EINN_PINN. Передаётся явно, чтобы не зашивать путь импорта.
        Пример: from NEW_PINN.PINN_const import EINN_PINN
    n_epoch : int
        Количество эпох обучения.
    lambda_data / lambda_ode / lambda_ic / lambda_bc : float
        Веса лосс-функций.
    device : str | None
        'cpu', 'cuda' или None (автоопределение).
    results_dir : str
        Папка для сохранения графиков.
    verbose : bool
        Выводить ли прогресс.
    """

    def __init__(
        self,
        pinn_class,
        n_epoch: int = 10_000,
        lambda_data: float = 0.01,
        lambda_ode: float = 1.0,
        lambda_ic: float = 0.1,
        lambda_bc: float = 0.1,
        device: Optional[str] = None,
        results_dir: str = "PINN_agent_results",
        verbose: bool = True,
    ):
        import torch
        self.pinn_class = pinn_class
        self.n_epoch = n_epoch
        self.lambda_data = lambda_data
        self.lambda_ode = lambda_ode
        self.lambda_ic = lambda_ic
        self.lambda_bc = lambda_bc
        self.results_dir = results_dir
        self.verbose = verbose
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

    # ─────────────────────────────────────────────────────────────
    # Публичный метод (аналог simulate у SurrogateAgent)
    # ─────────────────────────────────────────────────────────────

    def train(
        self,
        beta: float,
        gamma: float,
        mu: float,
        t: np.ndarray,
        S: np.ndarray,
        I: np.ndarray,
        R: np.ndarray,
        D: np.ndarray,
        population: float,
        train_size: int,
        tag: str = "run",
    ) -> Dict[str, Any]:
        """
        Обучает PINN с заданными параметрами.

        Returns
        -------
        dict:
            success       : bool
            final_params  : dict  — beta, gamma, mu после обучения
            plot_paths    : list  — пути к сохранённым графикам
            losses        : list  — история лосса по эпохам
            tag           : str
        """
        try:
            model = self._build_model(beta, gamma, mu, t, S, I, R, D, population, train_size)
            model.train_model(
                n_epoch=self.n_epoch,
                lambda_data=self.lambda_data,
                lambda_ode=self.lambda_ode,
                lambda_ic=self.lambda_ic,
                lambda_bc=self.lambda_bc,
            )
        except Exception as exc:
            return {"success": False, "error": str(exc), "tag": tag}

        S_pred, I_pred, R_pred, D_pred = model.predict(t)
        final_params = model.params.get_params_dict()

        plot_paths = self._save_plots(
            t, S, I, R, D,
            S_pred.numpy(), I_pred.numpy(), R_pred.numpy(), D_pred.numpy(),
            final_params, model.losses, train_size, tag,
        )

        return {
            "success": True,
            "final_params": {
                "beta":  float(final_params["beta"]),
                "gamma": float(final_params["gamma"]),
                "mu":    float(final_params["mu"]),
            },
            "plot_paths": plot_paths,
            "losses": model.losses,
            "tag": tag,
        }

    # ─────────────────────────────────────────────────────────────
    # Вызов из LangGraph (аналог __call__ у SurrogateAgent)
    # ─────────────────────────────────────────────────────────────

    def __call__(self, state: Dict) -> Dict:
        """
        Вызов агента из LangGraph pipeline.

        Читает параметры и данные из state, запускает train(),
        записывает результаты в state['pinn_results'].
        """
        if self.verbose:
            print("=" * 60)
            print("🧠 PINN AGENT")
            print("=" * 60)

        # 1. Получаем параметры
        params = self._get_params(state)

        if not params:
            if self.verbose:
                print("❌ Нет параметров для PINN")
            state["pinn_results"] = {
                "success": False,
                "error": (
                    "Нет параметров. Передайте generated_params "
                    "или убедитесь, что critic принял эпизод"
                ),
            }
            return state

        beta  = params["beta"]
        gamma = params["gamma"]
        mu    = params["mu"]

        if self.verbose:
            print(f"📐 Параметры: β={beta:.4f}, γ={gamma:.4f}, μ={mu:.5f}")

        # 2. Получаем данные
        data = self._get_data(state)

        if data is None:
            if self.verbose:
                print("❌ Нет данных для обучения PINN")
            state["pinn_results"] = {
                "success": False,
                "error": (
                    "Нет данных. Укажите pinn_data в state "
                    "или data_path в task_config"
                ),
            }
            return state

        t, S, I, R, D, population, train_size = data

        if self.verbose:
            print(f"📊 Данные: {len(t)} точек, train_size={train_size}")
            print(f"   S0={S[0]:.0f}, I0={I[0]:.0f}, R0={R[0]:.0f}, D0={D[0]:.0f}")
            print(f"   Популяция: {population:.0f}")
            print(f"🚀 Запуск обучения ({self.n_epoch} эпох на {self.device})...")

        # 3. Обучение
        tag = _run_tag(state.get("iteration", 0))
        results = self.train(
            beta=beta, gamma=gamma, mu=mu,
            t=t, S=S, I=I, R=R, D=D,
            population=population,
            train_size=train_size,
            tag=tag,
        )

        # 4. Вывод результатов
        if results.get("success"):
            fp = results["final_params"]
            if self.verbose:
                print("✅ Обучение завершено")
                print(f"   Финальные параметры:")
                print(f"      β={fp['beta']:.4f}, γ={fp['gamma']:.4f}, μ={fp['mu']:.5f}")
                print(f"   Графики сохранены: {results['plot_paths']}")
        else:
            if self.verbose:
                print(f"❌ Ошибка PINN: {results.get('error', 'неизвестная ошибка')}")

        state["pinn_results"] = results
        return state

    # ─────────────────────────────────────────────────────────────
    # Приватные методы
    # ─────────────────────────────────────────────────────────────

    def _get_params(self, state: Dict) -> Optional[Dict[str, float]]:
        """
        Возвращает dict {beta, gamma, mu}. Порядок приоритета:
          1. Последний принятый эпизод из history
          2. generated_params при critic_decision == 'accept'
          3. generated_params напрямую (запасной вариант)
        """
        history = state.get("history", [])
        accepted = [ep for ep in history if getattr(ep, "accepted", False)]
        if accepted:
            ep = accepted[-1]
            return {"beta": ep.beta, "gamma": ep.gamma, "mu": ep.mu}

        if state.get("critic_decision") == "accept":
            gp = state.get("generated_params", {})
            if gp:
                return gp

        gp = state.get("generated_params", {})
        if gp.get("beta"):
            return gp

        return None

    def _get_data(self, state: Dict):
        """
        Возвращает (t, S, I, R, D, population, train_size) или None.
        Источники (по приоритету):
        1. state['pinn_data']          — dict с готовыми массивами
        2. state['task_config']['pinn_data'] — dict внутри task_config
        3. task_config['data_path']    — путь к CSV-файлу
        """
        # Проверяем напрямую в state
        pinn_data = state.get("pinn_data")
        if pinn_data is not None:
            if self.verbose:
                print("📊 Использую pinn_data из state")
            return self._unpack_dict(pinn_data)
        
        # Проверяем в task_config
        task_config = state.get("task_config", {})
        pinn_data = task_config.get("pinn_data")
        if pinn_data is not None:
            if self.verbose:
                print("📊 Использую pinn_data из task_config")
            return self._unpack_dict(pinn_data)
        
        # Проверяем путь к файлу
        data_path = task_config.get("data_path")
        if data_path:
            if self.verbose:
                print(f"📂 Загружаю данные из {data_path}")
            return self._load_csv(data_path, task_config)
        
        if self.verbose:
            print("⚠️  Нет pinn_data в state и нет data_path в task_config")
            print(f"   Доступные ключи в state: {list(state.keys())}")
            print(f"   Доступные ключи в task_config: {list(task_config.keys())}")
        return None

    def _unpack_dict(self, pinn_data: dict):
        S = np.array(pinn_data["S"], dtype=float)
        I = np.array(pinn_data["I"], dtype=float)
        R = np.array(pinn_data["R"], dtype=float)
        D = np.array(pinn_data["D"], dtype=float)
        t = np.arange(len(S), dtype=float)
        population = float(pinn_data.get("population", S[0] + I[0] + R[0] + D[0]))
        train_size = int(pinn_data.get("train_size", int(len(S) * 0.75)))
        return t, S, I, R, D, population, train_size

    def _load_csv(self, path: str, task_config: dict):
        import pandas as pd
        try:
            df = pd.read_csv(path)
            S = df["S"].values.astype(float)
            I = df["I"].values.astype(float)
            R = df["R"].values.astype(float)
            D = df["D"].values.astype(float)
            t = np.arange(len(S), dtype=float)
            population = float(S[0] + I[0] + R[0] + D[0])
            train_size = int(task_config.get("train_size", int(len(S) * 0.75)))
            if self.verbose:
                print(f"📂 Данные загружены из {path}: {len(t)} точек, train_size={train_size}")
            return t, S, I, R, D, population, train_size
        except Exception as exc:
            if self.verbose:
                print(f"❌ Ошибка загрузки CSV {path}: {exc}")
            return None

    def _build_model(self, beta, gamma, mu, t, S, I, R, D, population, train_size):
        """
        Создаёт EINN_PINN и инициализирует latent-параметры
        значениями beta/gamma/mu от агента-генератора.
        """
        import torch

        model = self.pinn_class(
            t=t, S_data=S, I_data=I, R_data=R, D_data=D,
            population=population,
            train_size=train_size,
            device=str(self.device),
        )

        def to_logit(x: float) -> float:
            x = float(np.clip(x, 1e-12, 1 - 1e-12))
            return float(np.arctanh(2 * x - 1))

        with torch.no_grad():
            model.params.beta_latent.data  = torch.tensor(to_logit(beta),  dtype=torch.float32)
            model.params.gamma_latent.data = torch.tensor(to_logit(gamma), dtype=torch.float32)
            model.params.mu_latent.data    = torch.tensor(to_logit(mu),    dtype=torch.float32)

        # Включаем градиенты — PINN уточняет параметры в процессе обучения
        # model.params.beta_latent.requires_grad_(True)
        # model.params.gamma_latent.requires_grad_(True)
        # model.params.mu_latent.requires_grad_(True)

        # Пересобираем список оптимизируемых параметров
        model.all_params = (
            list(model.state_net.parameters()) +
            list(model.params.parameters())
        )

        return model

    # ─────────────────────────────────────────────────────────────
    # Сохранение графиков
    # ─────────────────────────────────────────────────────────────

    def _save_plots(
        self,
        t, S, I, R, D,
        S_pred, I_pred, R_pred, D_pred,
        final_params: dict,
        losses: list,
        train_size: int,
        tag: str,
    ) -> list:
        out_dir = _ensure_dir(self.results_dir)
        paths = []
        param_title = (
            f"β={final_params['beta']:.4f}, "
            f"γ={final_params['gamma']:.4f}, "
            f"μ={final_params['mu']:.5f}"
        )

        # ── 1. Общий fit (4 compartments) ──────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"PINN fit  |  {param_title}", fontsize=13)
        compartments = [
            ("S", S, S_pred, "steelblue"),
            ("I", I, I_pred, "tomato"),
            ("R", R, R_pred, "seagreen"),
            ("D", D, D_pred, "orchid"),
        ]
        for ax, (name, data, pred, color) in zip(axes.flat, compartments):
            ax.axvline(train_size, color="gray", lw=0.8, ls="--", label="train/test split")
            ax.plot(t, data, "o", ms=2, color=color, alpha=0.5, label="data")
            ax.plot(t, pred, "-", lw=1.8, color=color, label="PINN")
            ax.set_title(name)
            ax.set_xlabel("day")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        plt.tight_layout()
        fit_path = os.path.join(out_dir, f"{tag}_fit.png")
        fig.savefig(fit_path, dpi=120)
        plt.close(fig)
        paths.append(fit_path)

        # ── 2. Loss-кривая ──────────────────────────────────────
        if losses:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.semilogy(losses, lw=1.2, color="steelblue")
            ax.set_title("Training loss")
            ax.set_xlabel("epoch")
            ax.set_ylabel("loss (log scale)")
            ax.grid(alpha=0.3)
            loss_path = os.path.join(out_dir, f"{tag}_loss.png")
            fig.savefig(loss_path, dpi=120)
            plt.close(fig)
            paths.append(loss_path)

        # ── 3. Инфицированные крупно ────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.axvline(train_size, color="gray", lw=0.8, ls="--", label="train/test split")
        ax.plot(t, I,      "o", ms=2.5, color="tomato", alpha=0.6, label="I data")
        ax.plot(t, I_pred, "-", lw=2,   color="tomato",             label="I PINN")
        ax.set_title(f"Infected  |  {param_title}")
        ax.set_xlabel("day")
        ax.set_ylabel("count")
        ax.legend()
        ax.grid(alpha=0.3)
        inf_path = os.path.join(out_dir, f"{tag}_infected.png")
        fig.savefig(inf_path, dpi=120)
        plt.close(fig)
        paths.append(inf_path)

        return paths