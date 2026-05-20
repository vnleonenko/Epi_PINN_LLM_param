"""
main_test.py - Тестирование оптимизационного пайплайна с PINN валидацией
Сравнивает baseline и оптимизированные параметры через PINN
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import torch
import json 


# Добавляем пути для импортов
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.LLMFactory import LLMFactory
from agents.PINN_const import EINN_PINN
from agents.PINNAgent import PINNAgent
import config
from formats.data_formats import PipelineState
from agents.EpiParamGeneratorAgent import LLMEpiParamGenerator
from agents.DeterministicCriticAgent import DeterministicCriticAgent
from agents.IntentParserAgent import IntentParserAgent

from typing import Dict, Optional
from agents.BaseLLMClient import BaseLLMClient  
from typing import Union 

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import config


class SensitivityNode:
    """Node for computing parameter sensitivity map"""
    
    def __init__(self, surrogate_agent):
        self.surrogate_agent = surrogate_agent
    
    def __call__(self, state: PipelineState) -> PipelineState:
        print("=" * 80)
        print("📊 SENSITIVITY MAP COMPUTATION (EXTENDED)")
        print("=" * 80)
        
        # Получаем baseline параметры
        task_config = state.get('task_config', {})
        sim_params = state.get('simulation_params', {})
        
        beta = task_config.get('beta', 0.1)
        gamma = task_config.get('gamma', 0.05)
        mu = task_config.get('mu', 0.001)
        
        population = task_config.get('population', 10000)
        S0 = task_config.get('S0', 9999)
        I0 = task_config.get('I0', 1)
        R0 = task_config.get('R0', 0)
        D0 = task_config.get('D0', 0)
        
        t_max = sim_params.get('t_max', 200)
        num_points = sim_params.get('num_points', 1000)
        
        print(f"\n📍 BASELINE PARAMETERS:")
        print(f"   β = {beta:.6f}")
        print(f"   γ = {gamma:.6f}")
        print(f"   μ = {mu:.6f}")
        print(f"   Population: {population:,}, S0={S0:,}, I0={I0:,}")
        
        # Расширенные вариации
        variations_single = [-0.02, -0.01, 0.01, 0.02]   # 6 вариаций
        
        sensitivity = {
            'beta': {'variations': [], 'results': []},
            'gamma': {'variations': [], 'results': []},
            'mu': {'variations': [], 'results': []},
            'combined': {'variations': [], 'results': []},  # комбинированные изменения
            'baseline': {}
        }
        
        # Считаем baseline
        print("\n   Computing baseline...")
        baseline_result = self._simulate(
            beta, gamma, mu, 
            population, S0, I0, R0, D0,
            t_max, num_points
        )
        base_pos = baseline_result['peak_position']
        base_h = baseline_result['peak_height']
        
        sensitivity['baseline'] = {
            'beta': beta, 'gamma': gamma, 'mu': mu,
            'peak_position': base_pos,
            'peak_height': base_h
        }
        print(f"   Baseline: peak={base_pos:.1f}, height={base_h:.0f}")
        
        # ============================================================
        # 1. ОДИНОЧНЫЕ ВАРИАЦИИ (расширенные)
        # ============================================================
        
        # Beta sensitivity
        print("\n   Computing beta sensitivity (6 variations)...")
        for var in variations_single:
            beta_test = beta * (1 + var)
            result = self._simulate(
                beta_test, gamma, mu,
                population, S0, I0, R0, D0,
                t_max, num_points
            )
            if result:
                sensitivity['beta']['variations'].append(var)
                sensitivity['beta']['results'].append({
                    'beta': beta_test,
                    'gamma': gamma,
                    'mu': mu,
                    'peak_position': result['peak_position'],
                    'peak_height': result['peak_height'],
                    'position_delta': result['peak_position'] - base_pos,
                    'height_delta': result['peak_height'] - base_h
                })
        
        # Gamma sensitivity
        print("\n   Computing gamma sensitivity (6 variations)...")
        for var in variations_single:
            gamma_test = gamma * (1 + var)
            result = self._simulate(
                beta, gamma_test, mu,
                population, S0, I0, R0, D0,
                t_max, num_points
            )
            if result:
                sensitivity['gamma']['variations'].append(var)
                sensitivity['gamma']['results'].append({
                    'beta': beta,
                    'gamma': gamma_test,
                    'mu': mu,
                    'peak_position': result['peak_position'],
                    'peak_height': result['peak_height'],
                    'position_delta': result['peak_position'] - base_pos,
                    'height_delta': result['peak_height'] - base_h
                })
        
        # Mu sensitivity
        print("\n   Computing mu sensitivity (6 variations)...")
        for var in variations_single:
            mu_test = mu * (1 + var) if mu > 0 else 0.001 * (1 + var)
            result = self._simulate(
                beta, gamma, mu_test,
                population, S0, I0, R0, D0,
                t_max, num_points
            )
            if result:
                sensitivity['mu']['variations'].append(var)
                sensitivity['mu']['results'].append({
                    'beta': beta,
                    'gamma': gamma,
                    'mu': mu_test,
                    'peak_position': result['peak_position'],
                    'peak_height': result['peak_height'],
                    'position_delta': result['peak_position'] - base_pos,
                    'height_delta': result['peak_height'] - base_h
                })
        
        # ============================================================
        # 2. КОМБИНИРОВАННЫЕ ВАРИАЦИИ
        # ============================================================
        print("\n   Computing combined variations...")
        
        # Полезные комбинации
        combinations = [
            # (beta_var, gamma_var, mu_var, description)
            (+0.05, -0.05, 0.0, "β↑ γ↓"),
            (-0.05, +0.05, 0.0, "β↓ γ↑"),
            (+0.05, +0.05, 0.0, "β↑ γ↑"),
            (-0.05, -0.05, 0.0, "β↓ γ↓"),
            (+0.10, -0.10, 0.0, "β↑↑ γ↓↓"),
            (-0.10, +0.10, 0.0, "β↓↓ γ↑↑"),
            (+0.05, 0.0, +0.05, "β↑ μ↑"),
            (-0.05, 0.0, -0.05, "β↓ μ↓"),
        ]
        
        for beta_var, gamma_var, mu_var, desc in combinations:
            beta_test = beta * (1 + beta_var)
            gamma_test = gamma * (1 + gamma_var)
            mu_test = mu * (1 + mu_var) if mu > 0 else 0.001
            
            result = self._simulate(
                beta_test, gamma_test, mu_test,
                population, S0, I0, R0, D0,
                t_max, num_points
            )
            if result:
                sensitivity['combined']['variations'].append(desc)
                sensitivity['combined']['results'].append({
                    'beta': beta_test,
                    'gamma': gamma_test,
                    'mu': mu_test,
                    'peak_position': result['peak_position'],
                    'peak_height': result['peak_height'],
                    'position_delta': result['peak_position'] - base_pos,
                    'height_delta': result['peak_height'] - base_h
                })
                print(f"      {desc}: peak {result['peak_position']:.1f} (Δ{result['peak_position']-base_pos:+.1f}), "
                      f"height {result['peak_height']:.0f} (Δ{result['peak_height']-base_h:+.0f})")
        
        # ============================================================
        # КРАСИВЫЙ ВЫВОД
        # ============================================================
        print("\n" + "=" * 80)
        print("📈 EXTENDED SENSITIVITY MAP")
        print("=" * 80)
        print(f"\n   BASELINE: position = {base_pos:.1f} days, height = {base_h:.0f} infected")
        print(f"   Parameters: β={beta:.5f}, γ={gamma:.5f}, μ={mu:.6f}")
        
        # Таблица для Beta
        print("\n   " + "─" * 75)
        print("   β (infection rate) sensitivity:")
        print("   " + "─" * 75)
        print(f"   {'Change':<10} {'β value':<12} {'Peak day':<12} {'Δ day':<12} {'Peak height':<14} {'Δ height':<12} {'Δh/Δβ':<10}")
        print("   " + "─" * 75)
        for i, var in enumerate(sensitivity['beta']['variations']):
            res = sensitivity['beta']['results'][i]
            sensitivity_ratio = res['height_delta'] / (res['beta'] - beta) if res['beta'] != beta else 0
            print(f"   {var:+.0%}        {res['beta']:.5f}      {res['peak_position']:.1f}         {res['position_delta']:+.1f}         {res['peak_height']:.0f}           {res['height_delta']:+.0f}        {sensitivity_ratio:.0f}")
        
        # Таблица для Gamma
        print("\n   " + "─" * 75)
        print("   γ (recovery rate) sensitivity:")
        print("   " + "─" * 75)
        print(f"   {'Change':<10} {'γ value':<12} {'Peak day':<12} {'Δ day':<12} {'Peak height':<14} {'Δ height':<12} {'Δh/Δγ':<10}")
        print("   " + "─" * 75)
        for i, var in enumerate(sensitivity['gamma']['variations']):
            res = sensitivity['gamma']['results'][i]
            sensitivity_ratio = res['height_delta'] / (res['gamma'] - gamma) if res['gamma'] != gamma else 0
            print(f"   {var:+.0%}        {res['gamma']:.5f}      {res['peak_position']:.1f}         {res['position_delta']:+.1f}         {res['peak_height']:.0f}           {res['height_delta']:+.0f}        {sensitivity_ratio:.0f}")
        
        # Таблица для комбинированных
        print("\n   " + "─" * 75)
        print("   COMBINED variations:")
        print("   " + "─" * 75)
        print(f"   {'Pattern':<12} {'Peak day':<12} {'Δ day':<12} {'Peak height':<14} {'Δ height':<12}")
        print("   " + "─" * 75)
        for i, desc in enumerate(sensitivity['combined']['variations']):
            res = sensitivity['combined']['results'][i]
            print(f"   {desc:<12} {res['peak_position']:.1f}         {res['position_delta']:+.1f}         {res['peak_height']:.0f}           {res['height_delta']:+.0f}")
        
        # ============================================================
        # ИНТЕРПРЕТАЦИЯ
        # ============================================================
        print("\n   " + "─" * 75)
        print("   📋 INTERPRETATION & RULES OF THUMB:")
        print("   " + "─" * 75)
        
        # Анализ чувствительности
        beta_sensitivity_pos = sum([r['position_delta'] for r in sensitivity['beta']['results']]) / len(sensitivity['beta']['results'])
        beta_sensitivity_h = sum([r['height_delta'] for r in sensitivity['beta']['results']]) / len(sensitivity['beta']['results'])
        gamma_sensitivity_pos = sum([r['position_delta'] for r in sensitivity['gamma']['results']]) / len(sensitivity['gamma']['results'])
        gamma_sensitivity_h = sum([r['height_delta'] for r in sensitivity['gamma']['results']]) / len(sensitivity['gamma']['results'])
        
        print(f"\n   • β +10% → peak moves {'EARLIER' if beta_sensitivity_pos < 0 else 'LATER'} by {abs(beta_sensitivity_pos*2):.1f} days, height {'INCREASES' if beta_sensitivity_h > 0 else 'DECREASES'} by {abs(beta_sensitivity_h*2):.0f}")
        print(f"   • γ +10% → peak moves {'EARLIER' if gamma_sensitivity_pos < 0 else 'LATER'} by {abs(gamma_sensitivity_pos*2):.1f} days, height {'INCREASES' if gamma_sensitivity_h > 0 else 'DECREASES'} by {abs(gamma_sensitivity_h*2):.0f}")
        print(f"   • μ has minimal effect on peak (< 1 day), mainly affects total deaths")
        
    
        
        # Сохраняем в state
        state['sensitivity_map'] = sensitivity
        
        print("\n" + "=" * 80)
        print("✅ Extended sensitivity map computed and saved to state")
        print("=" * 80)
        
        return state
    
    def _simulate(self, beta, gamma, mu, population, S0, I0, R0, D0, t_max, num_points):
        """Вспомогательный метод для симуляции"""
        temp_state = {
            'generated_params': {'beta': beta, 'gamma': gamma, 'mu': mu},
            'initial_conditions': {
                'population': population, 'S0': S0, 'I0': I0, 'R0': R0, 'D0': D0
            },
            'simulation_params': {'t_max': t_max, 'num_points': num_points}
        }
        result_state = self.surrogate_agent(temp_state)
        return result_state.get('surrogate_results', {})
    
    def _simulate(self, beta, gamma, mu, population, S0, I0, R0, D0, t_max, num_points):
        """Вспомогательный метод для симуляции"""
        temp_state = {
            'generated_params': {'beta': beta, 'gamma': gamma, 'mu': mu},
            'initial_conditions': {
                'population': population, 'S0': S0, 'I0': I0, 'R0': R0, 'D0': D0
            },
            'simulation_params': {'t_max': t_max, 'num_points': num_points}
        }
        result_state = self.surrogate_agent(temp_state)
        return result_state.get('surrogate_results', {})

class SurrogateNode:
    """Node for surrogate model evaluation"""
    
    def __init__(self, surrogate_agent):
        self.surrogate_agent = surrogate_agent
    
    def __call__(self, state: PipelineState) -> PipelineState:
        print("=" * 60)
        print("📊 SURROGATE MODEL")
        print("=" * 60)
        
        # Получаем сгенерированные параметры
        params = state.get('generated_params', {})
        task_config = state.get('task_config', {})
        baseline_gamma = task_config.get('gamma')  # из Phase 1
        baseline_mu = task_config.get('mu')        # из Phase 
        
        # if baseline_gamma is not None and baseline_mu is not None:
        #     old_gamma = params.get('gamma')
        #     old_mu = params.get('mu')
        #     params['gamma'] = baseline_gamma
        #     params['mu'] = baseline_mu
        #     print(f"   🔄 Substituted γ: {old_gamma:.4f} → {baseline_gamma:.4f}")
        #     print(f"   🔄 Substituted μ: {old_mu:.5f} → {baseline_mu:.5f}")
        #     print(f"   ✅ Keeping β: {params['beta']:.4f}")
        #     state['generated_params'] = params
        if not params:
            print("❌ No parameters to evaluate")
            return state
        
        # Добавляем начальные условия в state, если их нет
        if 'initial_conditions' not in state:
            # Используем значения по умолчанию из config или из task_config
            task_config = state.get('task_config', {})
            state['initial_conditions'] = {
                'population': task_config.get('population', 10_000),
                'S0': task_config.get('S0', 9_999),
                'I0': task_config.get('I0', 1),
                'R0': task_config.get('R0', 0),
                'D0': task_config.get('D0', 0)
            }
        
        # Вызываем суррогатного агента
        state = self.surrogate_agent(state)
        
        # Результаты уже в state['surrogate_results']
        results = state.get('surrogate_results', {})
        
        if results.get('success', False):
            print(f"✅ Peak: {results['peak_position']:.1f} days")
            print(f"✅ Deaths: {results['total_deaths']:.0f}")
        else:
            print(f"❌ Simulation failed: {results.get('error', 'Unknown error')}")
        
        return state
    
class HistoryNode:
    """Node for managing history"""
    
    def __init__(self, generator=None, critic=None):
        """
        Initialize history node
        
        Args:
            generator: Generator agent to update its history
            critic: Critic agent to get history from
        """
        self.generator = generator
        self.critic = critic
    
    def __call__(self, state: PipelineState) -> PipelineState:
        print("=" * 60)
        print("📝 HISTORY MANAGER")
        print("=" * 60)
        
        # Get current iteration
        current_iteration = state.get('iteration', 0)
        
        # Get generated parameters and results
        generated_params = state.get('generated_params', {})
        surrogate_results = state.get('surrogate_results', {})
        critic_decision = state.get('critic_decision', 'reject')
        critic_reasoning = state.get('critic_reasoning', '')
        
        # Create Episode object with all available information
        from formats.data_formats import Episode
        
        episode = Episode(
            beta=generated_params.get('beta', 0.0),
            gamma=generated_params.get('gamma', 0.0),
            mu=generated_params.get('mu', 0.0),
            # Optional fields from surrogate results
            peak_position=surrogate_results.get('peak_position', 0.0),
            peak_height=surrogate_results.get('peak_height', 0.0),
            total_deaths=surrogate_results.get('total_deaths', 0.0),
            # Additional metadata
            iteration=current_iteration,
            expert_comment=state.get('expert_comment'),
            accepted=(critic_decision == 'accept'),
            reasoning=critic_reasoning
        )
        
        # Add to history in state
        history = state.get('history', [])
        history.append(episode)
        state['history'] = history
        
        # ✅ Update critic's history (critic stores episodes with evaluation results)
        if self.critic:
            # Проверяем, нет ли уже такого эпизода
            if not any(e.iteration == episode.iteration for e in self.critic.history):
                self.critic.add_to_history(episode)
        
        # ✅ Update generator's history (for context in next generations)
        if self.generator:
            # Generator needs history of all previous episodes for context
            # But doesn't need to add episodes itself - just reference critic's history
            # Synchronize generator's history with critic's history
            self.generator.history = self.critic.history
            print(f"✅ Synchronized generator history ({len(self.critic.history)} episodes)")
        
        # Always update current_episode for next iteration
        state['current_episode'] = episode
        
        if critic_decision == 'accept':
            print(f"✅ Episode {current_iteration} ACCEPTED and added to history")
        else:
            print(f"❌ Episode {current_iteration} REJECTED and added to history")
        
        print(f"📊 Total history: {len(history)} episodes")
        print(f"   - Accepted: {len([ep for ep in history if ep.accepted])}")
        print(f"   - Rejected: {len([ep for ep in history if not ep.accepted])}")
        
        return state


class DecisionNode:
    """Decision node for controlling the loop"""
    
    def __call__(self, state: PipelineState) -> str:
        print("=" * 60)
        print("⚖️ DECISION NODE")
        print("=" * 60)
        
        decision = state.get('critic_decision', 'reject')
        iteration = state.get('iteration', 0)
        max_iterations = state.get('max_iterations', 10)
        
        print(f"Iteration: {iteration}/{max_iterations}")
        print(f"Decision: {decision}")
        
        # Continue conditions
        if decision == 'accept':
            print("✅ Parameters accepted - ending loop")
            return "end"
        elif iteration >= max_iterations:
            print(f"⚠️ Max iterations ({max_iterations}) reached - ending loop")
            return "end"
        else:
            # 🔧 FIX: Increment iteration for the next loop
            state['iteration'] = iteration + 1
            print(f"🔄 Continuing loop with new generation (iteration {iteration + 1})")
            return "continue"
        

class PINNNode:
    """Node wrapper for PINN Agent in LangGraph"""
    
    def __init__(self, pinn_agent: PINNAgent):
        self.pinn_agent = pinn_agent
        self._original_n_epoch = pinn_agent.n_epoch  # запоминаем оригинал
    
    def __call__(self, state: PipelineState) -> PipelineState:
        print("=" * 60)
        print("🧠 PINN NODE")
        print("=" * 60)
        
        decision = state.get('critic_decision', 'reject')
        
        if decision == 'accept':
            print("✅ Parameters accepted, running PINN (validation mode: 1 epoch)...")
            
            # ✅ Временно ставим 1 эпоху
            self.pinn_agent.n_epoch = 1
            state = self.pinn_agent(state)
            # ✅ Возвращаем обратно
            self.pinn_agent.n_epoch = self._original_n_epoch
        else:
            print(f"⏭️ Parameters rejected, skipping PINN")
            state['pinn_results'] = {
                'success': False,
                'skipped': True,
                'reason': f'Parameters rejected by critic (decision: {decision})'
            }
        
        return state

class PINNVerificationNode:
    """
    Phase 3: Верификация на синтетических данных.
    Обучает PINN на данных от Surrogate (с β от LLM, γ,μ baseline).
    Параметры PINN LEARNABLE — проверяем, восстановит ли он параметры.
    """
    
    def __init__(self, n_epoch: int = 5000, device: str = None, save_plots: bool = True):
        self.n_epoch = n_epoch
        self.save_plots = save_plots
        import torch
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    
    def __call__(self, state: PipelineState) -> PipelineState:
        print("=" * 60)
        print("🔬 PINN VERIFICATION NODE (Phase 3)")
        print("=" * 60)
        
        # 1. Берем параметры от LLM (уже с подменой γ,μ)
        llm_params = state.get('generated_params', {})
        if not llm_params:
            print("❌ No LLM parameters")
            state['pinn_verification'] = {'success': False, 'error': 'No parameters'}
            return state
        
        print(f"📐 LLM parameters (after substitution):")
        print(f"   β = {llm_params.get('beta', 0):.4f}")
        print(f"   γ = {llm_params.get('gamma', 0):.4f}")
        print(f"   μ = {llm_params.get('mu', 0):.5f}")
        
        # 2. Получаем синтетические данные от Surrogate
        surrogate_results = state.get('surrogate_results', {})
        if not surrogate_results.get('success', False):
            print("❌ No surrogate results")
            state['pinn_verification'] = {'success': False, 'error': 'No surrogate data'}
            return state
        
        # Извлекаем траектории
        t = surrogate_results.get('t')
        I_synth = surrogate_results.get('I')
        S_synth = surrogate_results.get('S')
        R_synth = surrogate_results.get('R')
        D_synth = surrogate_results.get('D')
        
        if I_synth is None:
            print("❌ No trajectories in surrogate results")
            state['pinn_verification'] = {'success': False, 'error': 'No trajectories'}
            return state
        
        print(f"📊 Synthetic data: {len(t) if t is not None else 0} time points")
        print(f"   Synthetic peak: day {np.argmax(I_synth):.0f}, height {np.max(I_synth):.0f}")
        
        # 3. Начальные условия - берем из реальных данных
        task_config = state.get('task_config', {})
        pinn_data = task_config.get('pinn_data', {})
        
        if pinn_data:
            S_real = np.array(pinn_data.get('S', []))
            I_real = np.array(pinn_data.get('I', []))
            R_real = np.array(pinn_data.get('R', []))
            D_real = np.array(pinn_data.get('D', []))
            
            if len(S_real) > 0:
                population = float(S_real[0] + I_real[0] + R_real[0] + D_real[0])
                S0 = float(S_real[0])
                I0 = float(I_real[0])
                R0 = float(R_real[0])
                D0 = float(D_real[0])
                train_size = pinn_data.get('train_size')
                print(f"\n📊 Real data initial conditions (from pinn_data):")
                print(f"   Population: {population:.0f}")
                print(f"   S0={S0:.0f}, I0={I0:.0f}, R0={R0:.0f}, D0={D0:.0f}")
            else:
                initial_conditions = state.get('initial_conditions', {})
                population = initial_conditions.get('population', 10000)
                S0 = initial_conditions.get('S0', population - 100)
                I0 = initial_conditions.get('I0', 100)
                R0 = initial_conditions.get('R0', 0)
                D0 = initial_conditions.get('D0', 0)
                print(f"\n⚠️ Using fallback initial conditions")
        else:
            initial_conditions = state.get('initial_conditions', {})
            population = initial_conditions.get('population', 10000)
            S0 = initial_conditions.get('S0', population - 100)
            I0 = initial_conditions.get('I0', 100)
            R0 = initial_conditions.get('R0', 0)
            D0 = initial_conditions.get('D0', 0)
            print(f"\n⚠️ No pinn_data, using fallback initial conditions")
        
        # train_size = len(t) // 2
        
        # 4. Создаем и обучаем PINN
        from agents.PINN_const import EINN_PINN
        import torch
        import matplotlib.pyplot as plt
        
        print(f"\n🚀 Training PINN on synthetic data ({self.n_epoch} epochs)...")
        print(f"   Parameters: LEARNABLE, initialized with LLM values")
        
        # Конвертируем в numpy
        t_np = t.numpy() if hasattr(t, 'numpy') else np.array(t)
        S_np = S_synth.numpy() if hasattr(S_synth, 'numpy') else np.array(S_synth)
        I_np = I_synth.numpy() if hasattr(I_synth, 'numpy') else np.array(I_synth)
        R_np = R_synth.numpy() if hasattr(R_synth, 'numpy') else np.array(R_synth)
        D_np = D_synth.numpy() if hasattr(D_synth, 'numpy') else np.array(D_synth)

        print(f'train_size111 = {train_size}')
        
        model = EINN_PINN(
            t=t_np,
            S_data=S_np,
            I_data=I_np,
            R_data=R_np,
            D_data=D_np,
            population=population,
            train_size=len(t_np),
            device=str(self.device),
            init_params={
                'beta': float(llm_params['beta']),
                'gamma': float(llm_params['gamma']),
                'mu': float(llm_params['mu'])
            }
        )
        
        model.train_model(
            n_epoch=self.n_epoch,
            lambda_data=1.0,
            lambda_ode=0.1,
            lambda_ic=0.1,
            lambda_bc=0.0
        )
        
        # 5. Получаем восстановленные параметры и прогноз
        recovered = model.params.get_params_dict()
        S_pred, I_pred, R_pred, D_pred = model.predict(t_np)

        # Конвертируем предсказания в numpy
        I_pred_np = I_pred.numpy() if hasattr(I_pred, 'numpy') else np.array(I_pred)
        S_pred_np = S_pred.numpy() if hasattr(S_pred, 'numpy') else np.array(S_pred)
        R_pred_np = R_pred.numpy() if hasattr(R_pred, 'numpy') else np.array(R_pred)
        D_pred_np = D_pred.numpy() if hasattr(D_pred, 'numpy') else np.array(D_pred)

        # ============================================================
        # НОВОЕ: MC Dropout для оценки неопределенности
        # ============================================================
        print(f"\n📊 Running MC Dropout uncertainty estimation (100 passes, dropout_rate=0.05)...")
        uncertainty_results = model.predict_with_uncertainty(
            t_values=t_np, 
            n_passes=100, 
            dropout_rate=0.005
        )
        I_pred_np = uncertainty_results['mean']['I']
        S_pred_np = uncertainty_results['mean']['S']
        R_pred_np = uncertainty_results['mean']['R']
        D_pred_np = uncertainty_results['mean']['D']

        print(f"   Mean uncertainty (std) at peak: {np.mean(uncertainty_results['std']['I']):.1f}")
        
        print(f"✅ PINN recovered:")
        print(f"   β = {recovered['beta']:.4f} (Δ = {recovered['beta'] - llm_params['beta']:+.4f})")
        print(f"   γ = {recovered['gamma']:.4f} (Δ = {recovered['gamma'] - llm_params['gamma']:+.4f})")
        print(f"   μ = {recovered['mu']:.5f} (Δ = {recovered['mu'] - llm_params['mu']:+.5f})")
        
        # Анализ пиков
        synth_peak_idx = np.argmax(I_np)
        pinn_peak_idx = np.argmax(I_pred_np)
        
        synth_peak_day = t_np[synth_peak_idx]
        synth_peak_height = I_np[synth_peak_idx]
        pinn_peak_day = t_np[pinn_peak_idx]
        pinn_peak_height = I_pred_np[pinn_peak_idx]
        
        print(f"\n📊 Peak analysis:")
        print(f"   Synthetic data:  day {synth_peak_day:.0f}, height {synth_peak_height:.0f}")
        print(f"   PINN recovered:  day {pinn_peak_day:.0f}, height {pinn_peak_height:.0f}")
        print(f"   Peak day error:  {abs(pinn_peak_day - synth_peak_day):.1f} days")
        print(f"   Peak height error: {abs(pinn_peak_height - synth_peak_height):.0f} ({abs(pinn_peak_height - synth_peak_height)/synth_peak_height*100:.1f}%)")
        
        # 6. Сохраняем графики
        plot_paths = []
        if self.save_plots:
            os.makedirs("PINN_verification_plots", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            iteration = state.get('iteration', 0)
            
            # FIGURE 1: Сравнение Synthetic vs PINN (I(t))
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'PINN Verification (Iteration {iteration})\nLLM β={llm_params["beta"]:.4f} → PINN β={recovered["beta"]:.4f}', 
                        fontsize=14, fontweight='bold')
            
            # I(t)
            ax1 = axes[0, 0]
            ax1.plot(t_np, I_np, 'o', ms=3, color='blue', alpha=0.6, label='Synthetic data (SIRD)')
            ax1.plot(t_np, I_pred_np, '-', linewidth=2, color='red', alpha=0.8, label='PINN recovered')

            ax1.fill_between(
                t_np, 
                uncertainty_results['ci_lower_95']['I'], 
                uncertainty_results['ci_upper_95']['I'],
                alpha=0.25, color='red', label='95% CI'
            )
            ax1.axvline(train_size, color='gray', linestyle='--', alpha=0.5, label='Train/test split')
            ax1.scatter(synth_peak_day, synth_peak_height, color='blue', s=120 , marker='v', 
                    edgecolors='black', linewidth=1.5, zorder=5, label=f'Synthetic peak')
            ax1.scatter(pinn_peak_day, pinn_peak_height, color='red', s=120, marker='^', 
                    edgecolors='black', linewidth=1.5, zorder=5, label=f'PINN peak')
            ax1.set_xlabel('Days')
            ax1.set_ylabel('Infected (I)')
            ax1.set_title(f'I(t): Synthetic vs PINN Recovery')
            ax1.legend(loc='upper right', fontsize=6, ncol=2)
            ax1.grid(True, alpha=0.3)
            
            # I(t) log scale
            ax2 = axes[0, 1]
            ax2.semilogy(t_np, I_np + 1, 'o', ms=3, color='blue', alpha=0.6, label='Synthetic data')
            ax2.semilogy(t_np, I_pred_np + 1, '-', linewidth=2, color='red', alpha=0.8, label='PINN recovered')
            ax2.axvline(train_size, color='gray', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Days')
            ax2.set_ylabel('Infected (I) - log scale')
            ax2.set_title('I(t) Comparison (Log Scale)')
            ax2.legend(loc='upper right', fontsize=9)
            ax2.grid(True, alpha=0.3)
            
            # S(t)
            ax3 = axes[1, 0]
            ax3.plot(t_np, S_np, 'o', ms=3, color='green', alpha=0.6, label='Synthetic data')
            ax3.plot(t_np, S_pred_np, '-', linewidth=2, color='darkgreen', alpha=0.8, label='PINN recovered')
            ax3.axvline(train_size, color='gray', linestyle='--', alpha=0.5)
            ax3.set_xlabel('Days')
            ax3.set_ylabel('Susceptible (S)')
            ax3.set_title('S(t) Comparison')
            ax3.legend(loc='best', fontsize=9)
            ax3.grid(True, alpha=0.3)
            
            # D(t)
            ax4 = axes[1, 1]
            ax4.plot(t_np, D_np, 'o', ms=3, color='purple', alpha=0.6, label='Synthetic data')
            ax4.plot(t_np, D_pred_np, '-', linewidth=2, color='darkviolet', alpha=0.8, label='PINN recovered')
            ax4.axvline(train_size, color='gray', linestyle='--', alpha=0.5)
            ax4.set_xlabel('Days')
            ax4.set_ylabel('Deceased (D)')
            ax4.set_title('D(t) Comparison')
            ax4.legend(loc='best', fontsize=9)
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path1 = os.path.join("PINN_verification_plots", f"verification_{timestamp}_iter{iteration}.png")
            plt.savefig(plot_path1, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path1)
            print(f"   ✅ Plot saved: {plot_path1}")
            
            # FIGURE 2: Parameter recovery visualization
            fig2, ax = plt.subplots(figsize=(10, 6))
            
            params_names = ['β', 'γ', 'μ']
            llm_values = [llm_params['beta'], llm_params['gamma'], llm_params['mu']]
            pinn_values = [recovered['beta'], recovered['gamma'], recovered['mu']]
            
            x = np.arange(len(params_names))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, llm_values, width, label='LLM parameters', color='steelblue', alpha=0.8)
            bars2 = ax.bar(x + width/2, pinn_values, width, label='PINN recovered', color='coral', alpha=0.8)
            
            ax.set_ylabel('Parameter value')
            ax.set_title(f'Parameter Recovery: LLM vs PINN\nβ error: {recovered["beta"]-llm_params["beta"]:+.4f} '
                        f'({abs(recovered["beta"]-llm_params["beta"])/llm_params["beta"]*100:.1f}%)')
            ax.set_xticks(x)
            ax.set_xticklabels(params_names)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar, val in zip(bars1, llm_values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
            for bar, val in zip(bars2, pinn_values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            plot_path2 = os.path.join("PINN_verification_plots", f"parameters_{timestamp}_iter{iteration}.png")
            plt.savefig(plot_path2, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path2)
            print(f"   ✅ Plot saved: {plot_path2}")
        
        # 7. Сравнение
        comparison = {
            'success': True,
            'llm_params': dict(llm_params),
            'pinn_recovered_params': recovered,
            'predictions': {
                't': t_np.tolist(),
                'I': I_pred_np.tolist(),
                'S': S_pred_np.tolist(),
                'R': R_pred_np.tolist(),
                'D': D_pred_np.tolist()
            },
            'uncertainty': {
                'mean_I': uncertainty_results['mean']['I'].tolist(),
                'std_I': uncertainty_results['std']['I'].tolist(),
                'ci_lower_95_I': uncertainty_results['ci_lower_95']['I'].tolist(),
                'ci_upper_95_I': uncertainty_results['ci_upper_95']['I'].tolist(),
                'n_passes': uncertainty_results['n_passes']
            },
            'parameter_error': {
                'beta': recovered['beta'] - llm_params['beta'],
                'gamma': recovered['gamma'] - llm_params['gamma'],
                'mu': recovered['mu'] - llm_params['mu']
            },
            'relative_error': {
                'beta': abs(recovered['beta'] - llm_params['beta']) / llm_params['beta'] if llm_params['beta'] > 0 else 0,
                'gamma': abs(recovered['gamma'] - llm_params['gamma']) / llm_params['gamma'] if llm_params['gamma'] > 0 else 0,
                'mu': abs(recovered['mu'] - llm_params['mu']) / llm_params['mu'] if llm_params['mu'] > 0 else 0,
            },
            'peak_analysis': {
                'synthetic': {'day': float(synth_peak_day), 'height': float(synth_peak_height)},
                'pinn_recovered': {'day': float(pinn_peak_day), 'height': float(pinn_peak_height)},
                'error': {
                    'day': float(abs(pinn_peak_day - synth_peak_day)),
                    'height': float(abs(pinn_peak_height - synth_peak_height)),
                    'height_relative': float(abs(pinn_peak_height - synth_peak_height) / synth_peak_height) if synth_peak_height > 0 else 0
                }
            },
            'final_loss': model.losses[-1] if model.losses else None,
            'train_size': train_size,
            'plot_paths': plot_paths
        }
        
        print(f"\n📊 Recovery errors:")
        print(f"   Δβ = {comparison['parameter_error']['beta']:+.6f} ({comparison['relative_error']['beta']*100:.1f}%)")
        print(f"   Δγ = {comparison['parameter_error']['gamma']:+.6f} ({comparison['relative_error']['gamma']*100:.1f}%)")
        print(f"   Δμ = {comparison['parameter_error']['mu']:+.6f} ({comparison['relative_error']['mu']*100:.1f}%)")
        
        print(f"\n✅ Saving pinn_verification to state: success={comparison.get('success')}")
        print(f"   Keys: {list(comparison.keys())}")
        state['pinn_verification'] = comparison
        import json
        os.makedirs("PINN_verification_results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = f"PINN_verification_results/verification_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"   💾 Saved to disk: {json_path}")
        return state

# ============================================
# LangGraph Pipeline
# ============================================

class OptimizationPipeline:
    """
    Complete optimization pipeline using LangGraph
    """
    
    def __init__(
        self, 
        llm: Union[BaseLLMClient, object], 
        surrogate_agent=None, 
        initial_conditions=None,
        pinn_agent: Optional[PINNAgent] = None,  # Новый параметр
        use_pinn: bool = True,  # Флаг для включения/отключения PINN
        # critic_mode: str = "primary"
    ):
        self.llm = llm
        self.is_base_client = isinstance(llm, BaseLLMClient)
        self.use_pinn = use_pinn
        
        # Создаем суррогатного агента
        if surrogate_agent is None:
            from agents.SurrogateModel import SurrogateAgent
            self.surrogate_agent = SurrogateAgent(verbose=True)
        else:
            self.surrogate_agent = surrogate_agent
        
        # Создаем или используем переданный PINN агент
        if use_pinn:
            if pinn_agent is None:
                # Создаем PINN агент с параметрами по умолчанию
                self.pinn_agent = PINNAgent(
                    pinn_class=EINN_PINN,
                    n_epoch=10_000,
                    lambda_data=0.01,
                    lambda_ode=1.0,
                    results_dir="PINN_agent_results",
                    verbose=True
                )
            else:
                self.pinn_agent = pinn_agent
            self.pinn_node = PINNNode(self.pinn_agent)
        else:
            self.pinn_agent = None
            self.pinn_node = None
        
        # Сохраняем начальные условия для использования в pipeline
        self.initial_conditions = initial_conditions or {
            'population': 10_000,
            'S0': 9_999,
            'I0': 1,
            'R0': 0,
            'D0': 0
        }
        
        # Create agents
        llm_for_agents = self._get_llm_for_agents()

        self.intent_parser = IntentParserAgent(llm_for_agents)
        self.generator = LLMEpiParamGenerator(llm_for_agents, enable_logging=True, log_format="json")
        self.critic = DeterministicCriticAgent(
                                                llm_for_agents,
                                                enable_logging=True,
                                                log_format="json",
                                                max_retries=3,
                                                position_threshold=1.0,
                                                height_threshold_relative=0.03,
                                                position_tolerance=250.0,           # ±5 дней для "unchanged"
                                                height_tolerance_relative=0.03    # ±5% для "unchanged"
                                            )
        
        # Pass both generator and critic to history node
        self.history_node = HistoryNode(generator=self.generator, critic=self.critic)
        self.surrogate_node = SurrogateNode(self.surrogate_agent)
        self.decision_node = DecisionNode()
        self.memory = MemorySaver()
        
        # Build graph
        self.graph = self._build_graph()

    def _get_llm_for_agents(self):
        """Get LLM object compatible with agents (langchain format)"""
        if self.is_base_client:
            return self._create_langchain_compatible_llm()
        return self.llm
    
    def _create_langchain_compatible_llm(self):
        """Create wrapper for BaseLLMClient"""
        from langchain_core.language_models.llms import LLM
        from typing import Any, List, Mapping, Optional
        
        class LLMClientWrapper(LLM):
            client: BaseLLMClient
            
            @property
            def _llm_type(self) -> str:
                return "base_llm_client_wrapper"
            
            def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
                response = self.client.invoke(prompt)
                return response.content
            
            @property
            def _identifying_params(self) -> Mapping[str, Any]:
                return {"model": self.client.model_name}
            
            @property
            def temperature(self):
                return self.client.temperature
            
            @temperature.setter
            def temperature(self, value):
                self.client.temperature = value
        
        return LLMClientWrapper(client=self.llm)
    
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(PipelineState)

        print(f"\n🔧 Building graph: use_pinn={self.use_pinn}, pinn_agent={self.pinn_agent is not None}")
        
        sensitivity_node = SensitivityNode(self.surrogate_agent)
        
        # Add nodes
        workflow.add_node("sensitivity", sensitivity_node)
        workflow.add_node("intent", self.intent_parser)
        workflow.add_node("generate", self.generator.generate)
        workflow.add_node("surrogate", self.surrogate_node)
        workflow.add_node("critic", self.critic)
        workflow.add_node("history", self.history_node)
        
        # ✅ ДОБАВИТЬ: Создание узла pinn_verification
        if self.use_pinn and self.pinn_agent:
            workflow.add_node("pinn", self.pinn_node)  # старый PINN (если нужен)
            self.pinn_verification_node = PINNVerificationNode(n_epoch=10000, save_plots=True)
            workflow.add_node("pinn_verification", self.pinn_verification_node)  # ← ЭТО КРИТИЧЕСКИ ВАЖНО
        
        # Edges
        workflow.set_entry_point("sensitivity")
        workflow.add_edge("sensitivity", "intent")
        workflow.add_edge("intent", "generate")
        workflow.add_edge("generate", "surrogate")
        workflow.add_edge("surrogate", "critic")
        workflow.add_edge("critic", "history")
        
        # После history: conditional routing
        if self.use_pinn and self.pinn_agent:
            def route_after_history(state: PipelineState) -> str:
                decision = state.get('critic_decision', 'reject')
                iteration = state.get('iteration', 0)
                max_iterations = state.get('max_iterations', 10)
                
                print(f"\n🔄 ROUTING after history (iter {iteration}):")
                print(f"   Decision: {decision}")
                
                if decision == 'accept':
                    print(f"   ➡️  ACCEPTED → Running PINN VERIFICATION (Phase 3)")
                    return "pinn_verification"
                elif iteration >= max_iterations:
                    print(f"   ⏹️  Max iterations reached → END")
                    return "end"
                else:
                    print(f"   ➡️  REJECTED → Continue to next iteration")
                    state['iteration'] = iteration + 1
                    return "continue"
            
            workflow.add_conditional_edges(
                "history",
                route_after_history,
                {
                    "pinn_verification": "pinn_verification",
                    "continue": "generate",
                    "end": END
                }
            )
            
            # После верификации - конец
            workflow.add_edge("pinn_verification", END)
            
        else:
            # Без PINN: стандартная логика
            workflow.add_conditional_edges(
                "history",
                self.decision_node,
                {
                    "continue": "generate",
                    "end": END
                }
            )
        
        return workflow.compile(checkpointer=self.memory)
    
    def run(
    self,
    beta: float,
    gamma: float,
    mu: float,
    expert_comment: str = None,
    max_iterations: int = 10,
    t_max: int = 200,
    num_points: int = 1000,
    pinn_data: Dict = None,  # ← ОБЯЗАТЕЛЬНО содержит S, I, R, D
) -> Dict:
        """
        Run the optimization pipeline starting from baseline parameters
        
        Args:
            beta: Infection rate
            gamma: Recovery rate  
            mu: Mortality rate
            expert_comment: Expert guidance for optimization
            max_iterations: Maximum optimization iterations
            population, S0, I0, R0, D0: Initial conditions
            t_max, num_points: Simulation parameters
            pinn_data: Optional real data for PINN validation
        
        Returns:
            Final pipeline state
        """
        print("\n" + "=" * 60)
        print("🚀 STARTING OPTIMIZATION PIPELINE")
        print("=" * 60)
        print(f"📊 Baseline parameters: β={beta:.4f}, γ={gamma:.4f}, μ={mu:.5f}")
        if expert_comment:
            print(f"💬 Expert comment: {expert_comment}")
        # ============================================================
        # ✅ ИЗВЛЕКАЕМ НАЧАЛЬНЫЕ УСЛОВИЯ ИЗ РЕАЛЬНЫХ ДАННЫХ
        # ============================================================
        if pinn_data is None:
            raise ValueError("pinn_data is required! Must contain S, I, R, D arrays from real data.")
        
        S_real = np.array(pinn_data.get('S', []))
        I_real = np.array(pinn_data.get('I', []))
        R_real = np.array(pinn_data.get('R', []))
        D_real = np.array(pinn_data.get('D', []))
        
        if len(S_real) == 0:
            raise ValueError("pinn_data must contain non-empty S, I, R, D arrays")
        
        # Вычисляем начальные условия
        population = float(S_real[0] + I_real[0] + R_real[0] + D_real[0])
        S0 = float(S_real[0])
        I0 = float(I_real[0])
        R0 = float(R_real[0])
        D0 = float(D_real[0])
        
        print(f"\n📊 Real data initial conditions:")
        print(f"   Population: {population:.0f}")
        print(f"   S0={S0:.0f}, I0={I0:.0f}, R0={R0:.0f}, D0={D0:.0f}")
        # ============================================================
        # ШАГ 1: Получаем baseline через суррогат
        # ============================================================
        print("\n" + "=" * 60)
        print("📏 STEP 1: Computing baseline via surrogate")
        print("=" * 60)
        
        # Создаем временный state для baseline
        baseline_state = {
            'task_config': {
                'population': population,
                'S0': S0,
                'I0': I0,
                'R0': R0,
                'D0': D0,
                't_max': t_max,
                'num_points': num_points,
            },
            'generated_params': {'beta': beta, 'gamma': gamma, 'mu': mu},
            'initial_conditions': {
                'population': population,
                'S0': S0,
                'I0': I0,
                'R0': R0,
                'D0': D0
            }
        }
        
        # Запускаем суррогат
        baseline_state = self.surrogate_node(baseline_state)
        surrogate_results = baseline_state.get('surrogate_results', {})
        
        if not surrogate_results.get('success', False):
            raise RuntimeError(f"Failed to compute baseline: {surrogate_results.get('error')}")
        
        baseline_peak_position = surrogate_results['peak_position']
        baseline_peak_height = surrogate_results['peak_height']
        baseline_total_deaths = surrogate_results['total_deaths']
        
        print(f"\n✅ Baseline computed:")
        print(f"   Peak position: {baseline_peak_position:.1f} days")
        print(f"   Peak height: {baseline_peak_height:.0f} infected")
        print(f"   Total deaths: {baseline_total_deaths:.0f}")
        
        # ============================================================
        # ШАГ 2: Создаем начальный эпизод с baseline
        # ============================================================
        from formats.data_formats import Episode
        
        baseline_episode = Episode(
            beta=beta,
            gamma=gamma,
            mu=mu,
            peak_position=baseline_peak_position,
            peak_height=baseline_peak_height,
            total_deaths=baseline_total_deaths,
            iteration=0,
            expert_comment=f"BASELINE: {expert_comment if expert_comment else 'Initial parameters'}",
            accepted=True,  # Baseline всегда accepted
            reasoning="Baseline parameters from initial input"
        )
        
        # ============================================================
        # ШАГ 3: Инициализируем агентов с baseline
        # ============================================================
        self.critic.history = [baseline_episode]
        self.generator.history = [baseline_episode]
        print(f"\n📚 Initialized agents with baseline episode")
        
        # ============================================================
        # ШАГ 4: Формируем task_config для оптимизации
        # ============================================================
        task_config = {
            'description': f'Optimization from baseline: β={beta}, γ={gamma}, μ={mu}',
            'beta': beta,          
            'gamma': gamma,       
            'mu': mu,               
            'baseline_peak': baseline_peak_position,
            'baseline_height': baseline_peak_height,
            'baseline_deaths': baseline_total_deaths,
            'peak_tolerance': 5.0,
            'population': population,
            'S0': S0,
            'I0': I0,
            'R0': R0,
            'D0': D0,
            't_max': t_max,
            'num_points': num_points,
        }
        
        # Добавляем PINN данные если есть
        if pinn_data:
            task_config['pinn_data'] = pinn_data
        
        # ============================================================
        # ШАГ 5: Запускаем оптимизацию
        # ============================================================
        print("\n" + "=" * 60)
        print("🎯 STEP 2: Running optimization")
        print("=" * 60)
        
        self.critic.set_task_config(task_config)
        
        # Начальное состояние для оптимизации
        initial_state: PipelineState = {
            'task_config': task_config,
            'current_episode': baseline_episode,
            'expert_comment': expert_comment,
            'history': [baseline_episode],
            'generated_params': {'beta': beta, 'gamma': gamma, 'mu': mu},
            'surrogate_results': surrogate_results,
            'critic_decision': None,
            'critic_reasoning': None,
            'final_episode': None,
            'iteration': 0,
            'max_iterations': max_iterations,
            'should_continue': True,
            'initial_conditions': {
                'population': population,
                'S0': S0,
                'I0': I0,
                'R0': R0,
                'D0': D0
            },
            'simulation_params': {
                't_max': t_max,
                'num_points': num_points
            }
        }
        
        # Запускаем граф
        config = {"configurable": {"thread_id": "optimization_1"}}
        final_state = self.graph.invoke(initial_state, config)
        
        # ============================================================
        # ШАГ 6: Вывод результатов
        # ============================================================
        print("\n" + "=" * 60)
        print("🏁 PIPELINE COMPLETE")
        print("=" * 60)
        
        history = final_state.get('history', [])
        
        # Сравнение с baseline
        print(f"\n📊 Baseline (iteration 0):")
        print(f"   β={baseline_episode.beta:.4f}, γ={baseline_episode.gamma:.4f}, μ={baseline_episode.mu:.5f}")
        print(f"   Peak: {baseline_episode.peak_position:.1f} days, {baseline_episode.peak_height:.0f} infected")
        print(f"   Deaths: {baseline_episode.total_deaths:.0f}")
        
        # Лучший принятый эпизод
        accepted_episodes = [ep for ep in history if ep.accepted and ep.iteration > 0]
        if accepted_episodes:
            best = accepted_episodes[-1]
            print(f"\n✅ Best optimized (iteration {best.iteration}):")
            print(f"   β={best.beta:.4f} (Δ={best.beta - baseline_episode.beta:+.4f})")
            print(f"   γ={best.gamma:.4f} (Δ={best.gamma - baseline_episode.gamma:+.4f})")
            print(f"   μ={best.mu:.5f} (Δ={best.mu - baseline_episode.mu:+.5f})")
            print(f"   Peak: {best.peak_position:.1f} days (Δ={best.peak_position - baseline_episode.peak_position:+.1f})")
            print(f"   Deaths: {best.total_deaths:.0f} (Δ={best.total_deaths - baseline_episode.total_deaths:+.0f})")
            if best.reasoning:
                print(f"   Reasoning: {best.reasoning}")
        
        # ============================================================
        # ШАГ 6: Вывод результатов
        # ============================================================
        print("\n" + "=" * 60)
        print("🏁 PIPELINE COMPLETE")
        print("=" * 60)
        
        history = final_state.get('history', [])
        
        # ============================================================
        # ДЕТАЛЬНЫЙ ВЫВОД BASELINE
        # ============================================================
        print(f"\n📊 BASELINE (Initial Parameters):")
        print(f"   β={baseline_episode.beta:.4f}")
        print(f"   γ={baseline_episode.gamma:.4f}")
        print(f"   μ={baseline_episode.mu:.5f}")
        print(f"   📈 Peak position: {baseline_episode.peak_position:.1f} days")
        print(f"   📊 Peak height: {baseline_episode.peak_height:.0f} infected")
        print(f"   💀 Total deaths: {baseline_episode.total_deaths:.0f}")
        
        # ============================================================
        # ЛУЧШИЙ ОПТИМИЗИРОВАННЫЙ РЕЗУЛЬТАТ
        # ============================================================
        accepted_episodes = [ep for ep in history if ep.accepted and ep.iteration > 0]
        if accepted_episodes:
            best = accepted_episodes[-1]
            print(f"\n✅ BEST OPTIMIZED (Iteration {best.iteration}):")
            print(f"   β={best.beta:.4f} (Δ={best.beta - baseline_episode.beta:+.4f})")
            print(f"   γ={best.gamma:.4f} (Δ={best.gamma - baseline_episode.gamma:+.4f})")
            print(f"   μ={best.mu:.5f} (Δ={best.mu - baseline_episode.mu:+.5f})")
            print(f"   📈 Peak position: {best.peak_position:.1f} days (Δ={best.peak_position - baseline_episode.peak_position:+.1f})")
            print(f"   📊 Peak height: {best.peak_height:.0f} infected (Δ={best.peak_height - baseline_episode.peak_height:+.0f})")
            print(f"   💀 Total deaths: {best.total_deaths:.0f} (Δ={best.total_deaths - baseline_episode.total_deaths:+.0f})")
            if best.reasoning:
                print(f"   💭 Reasoning: {best.reasoning}")
        else:
            print(f"\n⚠️ No episodes were accepted during optimization")
        
        # ============================================================
        # ПОЛНАЯ ИСТОРИЯ ВСЕХ ИТЕРАЦИЙ
        # ============================================================
        print(f"\n📈 COMPLETE HISTORY ({len(history)} episodes):")
        print("-" * 100)
        print(f"{'Status':<6} {'Iter':<5} {'β':<8} {'γ':<8} {'μ':<10} {'Peak day':<10} {'Peak height':<12} {'Deaths':<10}")
        print("-" * 100)
        
        for ep in history:
            status = "✅" if ep.accepted else "❌"
            baseline_marker = " (BASELINE)" if ep.iteration == 0 else ""
            
            print(f"{status:<6} {ep.iteration:<5} "
                f"{ep.beta:<8.4f} {ep.gamma:<8.4f} {ep.mu:<10.5f} "
                f"{ep.peak_position:<10.1f} {ep.peak_height:<12.0f} {ep.total_deaths:<10.0f}"
                f"{baseline_marker}")
        
        print("-" * 100)
        
        # ============================================================
        # СТАТИСТИКА ПО ИТЕРАЦИЯМ
        # ============================================================
        if len(history) > 1:
            print(f"\n📊 STATISTICS:")
            print(f"   Total iterations: {len(history)}")
            print(f"   Accepted: {len([ep for ep in history if ep.accepted])}")
            print(f"   Rejected: {len([ep for ep in history if not ep.accepted])}")
            
            # Анализ тренда
            peaks = [ep.peak_position for ep in history if ep.iteration > 0]
            heights = [ep.peak_height for ep in history if ep.iteration > 0]
            deaths = [ep.total_deaths for ep in history if ep.iteration > 0]
            
            if peaks:
                print(f"\n📈 TRENDS (from baseline):")
                print(f"   Peak position: {baseline_episode.peak_position:.1f} → {peaks[-1]:.1f} days (Δ={peaks[-1] - baseline_episode.peak_position:+.1f})")
                print(f"   Peak height: {baseline_episode.peak_height:.0f} → {heights[-1]:.0f} infected (Δ={heights[-1] - baseline_episode.peak_height:+.0f})")
                print(f"   Total deaths: {baseline_episode.total_deaths:.0f} → {deaths[-1]:.0f} (Δ={deaths[-1] - baseline_episode.total_deaths:+.0f})")
        
        # ============================================================
        # РЕЗУЛЬТАТЫ PINN (если есть)
        # ============================================================
        pinn_results = final_state.get('pinn_results')
        if pinn_results and pinn_results.get('success'):
            print(f"\n🧠 PINN VALIDATION RESULTS:")
            fp = pinn_results.get('final_params', {})
            print(f"   Estimated parameters:")
            print(f"      β={fp.get('beta', 0):.4f}")
            print(f"      γ={fp.get('gamma', 0):.4f}")
            print(f"      μ={fp.get('mu', 0):.5f}")
            
            # Сравнение с оптимизированными параметрами
            if accepted_episodes:
                best = accepted_episodes[-1]
                print(f"\n   Difference from optimized:")
                print(f"      Δβ={fp.get('beta', 0) - best.beta:+.4f}")
                print(f"      Δγ={fp.get('gamma', 0) - best.gamma:+.4f}")
                print(f"      Δμ={fp.get('mu', 0) - best.mu:+.5f}")
            
            if pinn_results.get('plot_paths'):
                print(f"\n   📁 Plots saved: {len(pinn_results['plot_paths'])} files")
        
        return final_state



def visualize_pipeline(pipeline, filename="pipeline_graph.png"):
    """Визуализация графа пайплайна"""
    try:
        graph_png = pipeline.graph.get_graph(xray=True)
        png_bytes = graph_png.draw_mermaid_png()
        
        with open(filename, "wb") as f:
            f.write(png_bytes)
        
        print(f"✅ Pipeline visualization saved to {filename}")
        return True
    except Exception as e:
        print(f"❌ Error generating visualization: {e}")
        print("   Make sure you have installed: pip install matplotlib pillow")
        return False


def run_pinn_comparison(
    baseline_params: dict,
    optimized_params: dict,
    pinn_data: dict,
    pipeline_pinn_results: dict = None,
    results_dir: str = "PINN_comparison_results",
    n_epoch: int = 5_000,
    lambda_data: float = 1.0,
    lambda_ode: float = 1.0,
    lambda_ic: float = 0.1,
    lambda_bc: float = 0.0
):
    """
    Сравнивает baseline и оптимизированные параметры через PINN.
    Обучает PINN ровно 2 раза: baseline и (если нет в pipeline) optimized.
    """
    import matplotlib.pyplot as plt
    import json
    import torch
    
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "=" * 80)
    print("🧠 PINN COMPARISON: Baseline vs Optimized Predictions")
    print("=" * 80)
    
    # Подготовка данных
    S = np.array(pinn_data['S'])
    I = np.array(pinn_data['I'])
    R = np.array(pinn_data['R'])
    D = np.array(pinn_data['D'])
    t = np.arange(len(S), dtype=float)
    population = float(S[0] + I[0] + R[0] + D[0])
    train_size = pinn_data.get('train_size')
    
    from agents.PINN_const import EINN_PINN
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ============================================================
    # 1. ОБУЧЕНИЕ BASELINE (1-й раз)
    # ============================================================
    print("\n" + "-" * 60)
    print("📊 STEP 1: Training PINN for BASELINE parameters")
    print("-" * 60)
    print(f"   β={baseline_params['beta']:.4f}, γ={baseline_params['gamma']:.4f}, μ={baseline_params['mu']:.5f}")
    
    baseline_model = EINN_PINN(
        t=t, S_data=S, I_data=I, R_data=R, D_data=D,
        population=population, train_size=train_size,
        device=device, init_params=baseline_params
    )
    baseline_model.train_model(
        n_epoch=n_epoch,
        lambda_data=lambda_data,
        lambda_ode=lambda_ode,
        lambda_ic=lambda_ic,
        lambda_bc=lambda_bc
    )
    S_pred_bl, I_pred_bl, R_pred_bl, D_pred_bl = baseline_model.predict(t)
    I_pred_bl = I_pred_bl.numpy()
    S_pred_bl_np = S_pred_bl.numpy()
    R_pred_bl_np = R_pred_bl.numpy()
    D_pred_bl_np = D_pred_bl.numpy()
    baseline_est = baseline_model.params.get_params_dict()
    baseline_losses = baseline_model.losses.copy()
    
    print(f"   ✅ Baseline PINN complete")
    
    # ============================================================
    # 2. ПОЛУЧЕНИЕ OPTIMIZED (из pipeline или обучение - 2-й раз)
    # ============================================================
    print("\n" + "-" * 60)
    print("📊 STEP 2: Getting OPTIMIZED predictions")
    print("-" * 60)
    
    if pipeline_pinn_results and pipeline_pinn_results.get('success'):
        preds = pipeline_pinn_results.get('predictions')
        if preds is not None:
            print("   ✅ Using predictions from pipeline")
            I_pred_opt = np.array(preds['I'])
            S_pred_opt_np = np.array(preds['S'])
            R_pred_opt_np = np.array(preds['R'])
            D_pred_opt_np = np.array(preds['D'])
            optimized_est = pipeline_pinn_results.get('estimated_params', optimized_params)
            optimized_losses = pipeline_pinn_results.get('losses', [])
        else:
            print("   ⚠️ No predictions in pipeline, training...")
            optimized_model = EINN_PINN(
                t=t, S_data=S, I_data=I, R_data=R, D_data=D,
                population=population, train_size=train_size,
                device=device, init_params=optimized_params
            )
            optimized_model.train_model(
                n_epoch=n_epoch,
                lambda_data=lambda_data,
                lambda_ode=lambda_ode,
                lambda_ic=lambda_ic,
                lambda_bc=lambda_bc
            )
            S_pred_opt, I_pred_opt, R_pred_opt, D_pred_opt = optimized_model.predict(t)
            I_pred_opt = I_pred_opt.numpy()
            S_pred_opt_np = S_pred_opt.numpy()
            R_pred_opt_np = R_pred_opt.numpy()
            D_pred_opt_np = D_pred_opt.numpy()
            optimized_est = optimized_model.params.get_params_dict()
            optimized_losses = optimized_model.losses.copy()
            print(f"   ✅ Optimized model trained")
    else:
        print(f"   🔧 Training optimized model...")
        print(f"   β={optimized_params['beta']:.4f}, γ={optimized_params['gamma']:.4f}, μ={optimized_params['mu']:.5f}")
        optimized_model = EINN_PINN(
            t=t, S_data=S, I_data=I, R_data=R, D_data=D,
            population=population, train_size=train_size,
            device=device, init_params=optimized_params
        )
        optimized_model.train_model(
            n_epoch=n_epoch,
            lambda_data=lambda_data,
            lambda_ode=lambda_ode,
            lambda_ic=lambda_ic,
            lambda_bc=lambda_bc
        )
        S_pred_opt, I_pred_opt, R_pred_opt, D_pred_opt = optimized_model.predict(t)
        I_pred_opt = I_pred_opt.numpy()
        S_pred_opt_np = S_pred_opt.numpy()
        R_pred_opt_np = R_pred_opt.numpy()
        D_pred_opt_np = D_pred_opt.numpy()
        optimized_est = optimized_model.params.get_params_dict()
        optimized_losses = optimized_model.losses.copy()
        print(f"   ✅ Optimized model trained")
    
    # ============================================================
    # 3. Анализ пиков в I
    # ============================================================
    print("\n" + "-" * 60)
    print("📊 STEP 3: Analyzing I(t) peaks")
    print("-" * 60)
    
    baseline_peak_idx = np.argmax(I_pred_bl)
    optimized_peak_idx = np.argmax(I_pred_opt)
    data_peak_idx = np.argmax(I)
    
    baseline_peak_day = t[baseline_peak_idx]
    baseline_peak_height = I_pred_bl[baseline_peak_idx]
    optimized_peak_day = t[optimized_peak_idx]
    optimized_peak_height = I_pred_opt[optimized_peak_idx]
    data_peak_day = t[data_peak_idx]
    data_peak_height = I[data_peak_idx]
    
    peak_day_change = optimized_peak_day - baseline_peak_day
    peak_height_change = optimized_peak_height - baseline_peak_height
    peak_day_error_change = abs(optimized_peak_day - data_peak_day) - abs(baseline_peak_day - data_peak_day)
    
    print(f"\n   PEAK ANALYSIS:")
    print(f"   {'':<20} {'Day':<10} {'Height':<12}")
    print(f"   {'-'*42}")
    print(f"   {'Real data':<20} {data_peak_day:<10.1f} {data_peak_height:<12.0f}")
    print(f"   {'Baseline PINN':<20} {baseline_peak_day:<10.1f} {baseline_peak_height:<12.0f}")
    print(f"   {'Optimized PINN':<20} {optimized_peak_day:<10.1f} {optimized_peak_height:<12.0f}")
    
    print(f"\n   CHANGES (Optimized - Baseline):")
    print(f"   Peak day: {peak_day_change:+.1f} days")
    print(f"   Peak height: {peak_height_change:+.0f} infected")
    print(f"   Error to real data (day): {peak_day_error_change:+.1f} days")
    
    # ============================================================
    # 4. Сравнение параметров
    # ============================================================
    print("\n" + "-" * 60)
    print("📊 STEP 4: Parameter comparison")
    print("-" * 60)
    
    print(f"\n   {'Parameter':<10} {'Baseline Input':<15} {'Baseline Est':<15} {'Optimized Input':<16} {'Optimized Est':<15}")
    print(f"   {'-'*75}")
    for param, name in [('beta', 'β'), ('gamma', 'γ'), ('mu', 'μ')]:
        print(f"   {name:<10} {baseline_params[param]:<15.4f} {baseline_est[param]:<15.4f} "
              f"{optimized_params[param]:<16.4f} {optimized_est[param]:<15.4f}")
    
    # ============================================================
    # 5. Графики
    # ============================================================
    print("\n" + "-" * 60)
    print("📊 STEP 5: Creating comparison plots")
    print("-" * 60)
    
    # FIGURE 1: Сравнение I(t)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # I(t)
    ax1 = axes[0, 0]
    ax1.plot(t, I, 'o', ms=2, color='gray', alpha=0.5, label='Real data')
    ax1.plot(t, I_pred_bl, '-', linewidth=2, color='blue', alpha=0.8, label='Baseline PINN')
    ax1.plot(t, I_pred_opt, '-', linewidth=2, color='red', alpha=0.8, label='Optimized PINN')
    ax1.axvline(train_size, color='gray', linestyle='--', alpha=0.5, label='Train/test split')
    ax1.scatter(baseline_peak_day, baseline_peak_height, color='blue', s=100, marker='v', edgecolors='black', zorder=5)
    ax1.scatter(optimized_peak_day, optimized_peak_height, color='red', s=100, marker='^', edgecolors='black', zorder=5)
    ax1.scatter(data_peak_day, data_peak_height, color='gray', s=100, marker='s', edgecolors='black', zorder=5)
    ax1.set_xlabel('Days')
    ax1.set_ylabel('Infected (I)')
    ax1.set_title(f'I(t) Comparison\nPeak change: {peak_day_change:+.0f} days, {peak_height_change:+.0f} infected')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # I(t) log
    ax2 = axes[0, 1]
    ax2.semilogy(t, I + 1, 'o', ms=2, color='gray', alpha=0.5, label='Real data')
    ax2.semilogy(t, I_pred_bl + 1, '-', linewidth=2, color='blue', alpha=0.8, label='Baseline PINN')
    ax2.semilogy(t, I_pred_opt + 1, '-', linewidth=2, color='red', alpha=0.8, label='Optimized PINN')
    ax2.axvline(train_size, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Days')
    ax2.set_ylabel('Infected (I) - log scale')
    ax2.set_title('I(t) Comparison (Log Scale)')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # S(t)
    ax3 = axes[1, 0]
    ax3.plot(t, S, 'o', ms=2, color='gray', alpha=0.5, label='Real data')
    ax3.plot(t, S_pred_bl_np, '-', linewidth=2, color='blue', alpha=0.8, label='Baseline PINN')
    ax3.plot(t, S_pred_opt_np, '-', linewidth=2, color='red', alpha=0.8, label='Optimized PINN')
    ax3.axvline(train_size, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Days')
    ax3.set_ylabel('Susceptible (S)')
    ax3.set_title('S(t) Comparison')
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # D(t)
    ax4 = axes[1, 1]
    ax4.plot(t, D, 'o', ms=2, color='gray', alpha=0.5, label='Real data')
    ax4.plot(t, D_pred_bl_np, '-', linewidth=2, color='blue', alpha=0.8, label='Baseline PINN')
    ax4.plot(t, D_pred_opt_np, '-', linewidth=2, color='red', alpha=0.8, label='Optimized PINN')
    ax4.axvline(train_size, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Days')
    ax4.set_ylabel('Deceased (D)')
    ax4.set_title('D(t) Comparison')
    ax4.legend(loc='best', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    comparison_plot_path = os.path.join(results_dir, f"comparison_{timestamp}.png")
    plt.savefig(comparison_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Comparison plot saved: {comparison_plot_path}")
    
    # FIGURE 2: Детальный анализ I(t)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t, I, 'o', ms=2, color='gray', alpha=0.4, label='Real data')
    ax.plot(t, I_pred_bl, '-', linewidth=2.5, color='blue', alpha=0.7, label='Baseline PINN')
    ax.plot(t, I_pred_opt, '-', linewidth=2.5, color='red', alpha=0.7, label='Optimized PINN')
    ax.axvline(train_size, color='gray', linestyle='--', alpha=0.5)
    ax.scatter(baseline_peak_day, baseline_peak_height, color='blue', s=150, marker='v', edgecolors='black', linewidth=2, zorder=5)
    ax.scatter(optimized_peak_day, optimized_peak_height, color='red', s=150, marker='^', edgecolors='black', linewidth=2, zorder=5)
    ax.scatter(data_peak_day, data_peak_height, color='gray', s=150, marker='s', edgecolors='black', linewidth=2, zorder=5)
    # ax.annotate('', xy=(optimized_peak_day, optimized_peak_height), xytext=(baseline_peak_day, baseline_peak_height),
    #             arrowprops=dict(arrowstyle='->', color='green', lw=2))
    mid_x = (baseline_peak_day + optimized_peak_day) / 2
    mid_y = (baseline_peak_height + optimized_peak_height) / 2
    ax.text(0.98, 0.98, 
        f'Δ day: {peak_day_change:+.0f} days\nΔ height: {peak_height_change:+.0f} infected', 
        fontsize=11, ha='right', va='top', 
        transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='orange', alpha=0.9))
    ax.set_xlabel('Days', fontsize=12)
    ax.set_ylabel('Infected (I)', fontsize=12)
    ax.set_title(f'Peak Analysis: Baseline → Optimized', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    peak_analysis_path = os.path.join(results_dir, f"peak_analysis_{timestamp}.png")
    plt.savefig(peak_analysis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Peak analysis plot saved: {peak_analysis_path}")
    
     # ============================================================
    # 6. Сохранение результатов
    # ============================================================
    results = {
        'timestamp': timestamp,
        'baseline_params_input': baseline_params,
        'optimized_params_input': optimized_params,
        'baseline_pinn_estimated': baseline_est,
        'optimized_pinn_estimated': optimized_est,
        'baseline_pinn': {'success': True, 'losses': baseline_losses},
        'optimized_pinn': {'success': True, 'losses': optimized_losses},
        'peak_analysis': {
            'real_data': {'day': float(data_peak_day), 'height': float(data_peak_height)},
            'baseline': {'day': float(baseline_peak_day), 'height': float(baseline_peak_height)},
            'optimized': {'day': float(optimized_peak_day), 'height': float(optimized_peak_height)},
            'changes': {
                'peak_day': float(peak_day_change),
                'peak_height': float(peak_height_change),
                'error_to_real_day': float(peak_day_error_change)
            }
        },
        'plots': [comparison_plot_path, peak_analysis_path]
    }
    
    json_path = os.path.join(results_dir, f"comparison_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   ✅ Results saved: {json_path}")
    
    # ============================================================
    # 7. Итоговый вывод
    # ============================================================
    print("\n" + "=" * 80)
    print("📋 PINN VALIDATION SUMMARY")
    print("=" * 80)
    print(f"\n   Baseline peak:  day {baseline_peak_day:.0f}, height {baseline_peak_height:.0f}")
    print(f"   Optimized peak: day {optimized_peak_day:.0f}, height {optimized_peak_height:.0f}")
    print(f"   Real data peak: day {data_peak_day:.0f}, height {data_peak_height:.0f}")
    print(f"\n   Peak day change: {peak_day_change:+.0f} days")
    print(f"   Peak height change: {peak_height_change:+.0f} infected")

    results['t_data'] = t.tolist()
    results['I_real'] = I.tolist()
    results['i_predictions'] = {
        'baseline': I_pred_bl.tolist(),
        'optimized': I_pred_opt.tolist()
    }
    results['train_size'] = train_size
    
    return results


def main():
    """Основная функция тестирования"""
    
    print("=" * 80)
    print("🚀 OPTIMIZATION PIPELINE TEST WITH PINN VALIDATION")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ============================================================
    # 1. Инициализация LLM
    # ============================================================
    print("\n" + "-" * 60)
    print("📌 STEP 1: Initializing LLM")
    print("-" * 60)
    
    llm_client = LLMFactory.from_config(config.LLM_CONFIG)
    print(f"   ✅ LLM initialized: {llm_client.get_model_info()}")
    
    # ============================================================
    # 2. Загрузка данных для PINN
    # ============================================================
    print("\n" + "-" * 60)
    print("📌 STEP 2: Loading PINN data")
    print("-" * 60)

    USE_SYNTHETIC_DATA = True
    print(f"📌 Data mode: {'SYNTHETIC' if USE_SYNTHETIC_DATA else 'REAL'}")
    
    # covid_cases = pd.read_csv('../../NEW_PINN/real_datasets/covid-19_Kouprianov.csv')
    # covid_cases = pd.read_csv('../../NEW_PINN/real_datasets/PINN-COVID-Italy.csv')
    covid_cases = pd.read_csv('../../NEW_PINN/synthetic_datasets/01_baseline_constant.csv')
    
    # covid_cases = pd.read_csv('../../NEW_PINN/synthetic_datasets/05_complex_beta_gamma_dynamics_noise_3pct_seed42.csv')
    print(f"   ✅ Loaded {len(covid_cases)} data points")
    
    pinn_data = {
        'S': covid_cases['S'].tolist(),
        'I': covid_cases['I'].tolist(),
        'R': covid_cases['R'].tolist(),
        'D': covid_cases['D'].tolist(),
        'train_size': 120,
    }
    
    # ============================================================
    # 3. Создание PINN агента
    # ============================================================
    print("\n" + "-" * 60)
    print("📌 STEP 3: Creating PINN agent")
    print("-" * 60)

    # ЕДИНЫЕ ПАРАМЕТРЫ ОБУЧЕНИЯ
    n_epoch = 10000
    lambda_data = 1.0
    lambda_ode = 1.0
    lambda_ic = 0.1
    lambda_bc = 0.0

    pinn_agent = PINNAgent(
        pinn_class=EINN_PINN,
        n_epoch=n_epoch,
        lambda_data=lambda_data,
        lambda_ode=lambda_ode,
        lambda_ic=lambda_ic,
        lambda_bc=lambda_bc,
        results_dir="PINN_agent_results",
        verbose=True
    )
    print(f"   ✅ PINN agent created (device: {pinn_agent.device})")
    print(f"   Training config: epochs={n_epoch}, λ_data={lambda_data}, λ_ode={lambda_ode}")
    
    # ============================================================
    # 4. Создание пайплайна
    # ============================================================
    print("\n" + "-" * 60)
    print("📌 STEP 4: Creating optimization pipeline")
    print("-" * 60)
    
    pipeline = OptimizationPipeline(
        llm=llm_client,
        pinn_agent=pinn_agent,
        use_pinn=True,
        # critic_mode="both"
    )
    print("   ✅ Pipeline created")
    
    # ============================================================
    # 5. Визуализация пайплайна
    # ============================================================
    print("\n" + "-" * 60)
    print("📌 STEP 5: Visualizing pipeline")
    print("-" * 60)
    
    visualize_pipeline(pipeline, "pipeline_graph.png")
    
    # ============================================================
    # 6. Параметры для тестирования
    # ============================================================
    # baseline_beta = 0.1219
    # baseline_gamma = 0.099
    # baseline_mu = 0.0099

    baseline_beta = 0.091
    baseline_gamma = 0.0553
    baseline_mu = 0.0085
    
    # expert_comment = "The epidemic should end sooner"
    # expert_comment = "Need higher peak, position is unimportant"
    expert_comment = "Need higher peak"
    # expert_comment = "Need lower peak, position is unimportant"
    # expert_comment = "Mask mandate will be introduced"
    # expert_comment = "The peak should be higher and later"
    # expert_comment = "First move the peak LATER. Hieght is unimportant"
    # expert_comment =  "The infection rate graph shows an excessively sharp decline after the peak – in reality, the epidemic's 'tail' should be longer."
    # expert_comment = "Quarantine measures were introduced late – the peak should shift by 7-10 days."

    print("\n" + "-" * 60)
    print("📌 STEP 6: Test parameters")
    print("-" * 60)
    print(f"   Baseline: β={baseline_beta:.4f}, γ={baseline_gamma:.4f}, μ={baseline_mu:.5f}")
    print(f"   Expert comment: {expert_comment}")
    print(f"   Max iterations: 5")
    
    # ============================================================
    # 7. Запуск оптимизации
    # ============================================================
    print("\n" + "-" * 60)
    print("📌 STEP 7: Running optimization pipeline")
    print("-" * 60)
    
    result = pipeline.run(
        beta=baseline_beta,
        gamma=baseline_gamma,
        mu=baseline_mu,
        expert_comment=expert_comment,
        max_iterations=10,
        t_max=400,
        pinn_data=pinn_data
    )
    
    # ============================================================
    # 8. Извлечение результатов
    # ============================================================
    history = result.get('history', [])
    baseline_episode = next((ep for ep in history if ep.iteration == 0), None)
    
    accepted_episodes = [ep for ep in history if ep.accepted and ep.iteration > 0]
    optimized_episode = accepted_episodes[-1] if accepted_episodes else None
    
    if not baseline_episode:
        print("❌ Baseline episode not found!")
        return
    
    if not optimized_episode:
        print("⚠️ No optimized episode accepted, using last episode")
        optimized_episode = history[-1]
    
    # ============================================================
    # 9. Сравнение через PINN
    # ============================================================
    print("\n" + "-" * 60)
    print("📌 STEP 8: Running PINN comparison")
    print("-" * 60)
    
    baseline_params = {
        'beta': baseline_episode.beta,
        'gamma': baseline_episode.gamma,
        'mu': baseline_episode.mu
    }
    
    optimized_params = {
        'beta': optimized_episode.beta,
        'gamma': optimized_episode.gamma,
        'mu': optimized_episode.mu
    }
    
    # comparison_results = run_pinn_comparison(
    #     baseline_params=baseline_params,
    #     optimized_params=optimized_params,
    #     pinn_data=pinn_data,
    #     pipeline_pinn_results=result.get('pinn_results'),
    #     results_dir="PINN_comparison_results",
    #     # Передаем параметры обучения
    #     n_epoch=n_epoch,
    #     lambda_data=lambda_data,
    #     lambda_ode=lambda_ode,
    #     lambda_ic=lambda_ic,
    #     lambda_bc=lambda_bc
    # )
    if USE_SYNTHETIC_DATA:
        comparison_results = run_pinn_comparison(
            baseline_params=baseline_params,
            optimized_params=optimized_params,
            pinn_data=pinn_data,
            pipeline_pinn_results=result.get('pinn_results'),
            results_dir="PINN_comparison_results",
            n_epoch=n_epoch,
            lambda_data=lambda_data,
            lambda_ode=lambda_ode,
            lambda_ic=lambda_ic,
            lambda_bc=lambda_bc
        )
    else:
        # Для реальных данных: baseline = SIRD, optimized = SIRD
        print("   ℹ️ Using SIRD as baseline (not PINN) for real data")
        
        from scipy.integrate import odeint
        
        def simulate_sird(beta, gamma, mu, N, S0, I0, R0, D0, t_max=400, num_points=1000):
            t = np.linspace(0, t_max, num_points)
            y0 = [S0, I0, R0, D0]
            
            def sird_model(y, t, beta, gamma, mu, N):
                S, I, R, D = y
                dS = -beta * S * I / N
                dI = beta * S * I / N - gamma * I - mu * I
                dR = gamma * I
                dD = mu * I
                return [dS, dI, dR, dD]
            
            solution = odeint(sird_model, y0, t, args=(beta, gamma, mu, N))
            S, I, R, D = solution.T
            return {'t': t, 'I': I, 'S': S, 'R': R, 'D': D,
                    'peak_position': t[np.argmax(I)], 'peak_height': np.max(I)}
        
        # Получаем начальные условия
        S0 = pinn_data['S'][0]
        I0 = pinn_data['I'][0]
        R0 = pinn_data['R'][0]
        D0 = pinn_data['D'][0]
        population = S0 + I0 + R0 + D0
        
        # Baseline SIRD
        baseline_sird = simulate_sird(
            beta=baseline_params['beta'],
            gamma=baseline_params['gamma'],
            mu=baseline_params['mu'],
            N=population, S0=S0, I0=I0, R0=R0, D0=D0,
            t_max=400, num_points=1000
        )
        
        # Optimized SIRD
        optimized_sird = simulate_sird(
            beta=optimized_params['beta'],
            gamma=optimized_params['gamma'],
            mu=optimized_params['mu'],
            N=population, S0=S0, I0=I0, R0=R0, D0=D0,
            t_max=400, num_points=1000
        )
        
        # ✅ Для графика нужно интерполировать реальные данные на время симуляции
        t_sim = baseline_sird['t']  # 1000 точек
        I_real = np.array(pinn_data['I'])
        t_real = np.arange(len(I_real))
        
        # Интерполяция реальных данных на время симуляции
        from scipy.interpolate import interp1d
        interp_func = interp1d(t_real, I_real, kind='linear', fill_value='extrapolate')
        I_real_interp = interp_func(t_sim)
        
        # Находим пики (на интерполированных данных)
        baseline_peak_idx = np.argmax(baseline_sird['I'])
        optimized_peak_idx = np.argmax(optimized_sird['I'])
        data_peak_idx = np.argmax(I_real_interp)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        comparison_results = {
            'timestamp': timestamp,
            'baseline_params_input': baseline_params,
            'optimized_params_input': optimized_params,
            'baseline_pinn_estimated': baseline_params,
            'optimized_pinn_estimated': optimized_params,
            'baseline_pinn': {'success': True, 'losses': []},
            'optimized_pinn': {'success': True, 'losses': []},
            't_data': t_sim.tolist(),  # 1000 точек
            'I_real': I_real_interp.tolist(),  # ← интерполированные до 1000 точек
            'i_predictions': {
                'baseline': baseline_sird['I'].tolist(),
                'optimized': optimized_sird['I'].tolist()
            },
            'train_size': pinn_data.get('train_size', 120),
            'peak_analysis': {
                'real_data': {'day': float(t_sim[data_peak_idx]), 'height': float(I_real_interp[data_peak_idx])},
                'baseline': {'day': float(baseline_sird['t'][baseline_peak_idx]), 
                        'height': float(baseline_sird['I'][baseline_peak_idx])},
                'optimized': {'day': float(optimized_sird['t'][optimized_peak_idx]), 
                            'height': float(optimized_sird['I'][optimized_peak_idx])},
                'changes': {
                    'peak_day': float(optimized_sird['t'][optimized_peak_idx] - baseline_sird['t'][baseline_peak_idx]),
                    'peak_height': float(optimized_sird['I'][optimized_peak_idx] - baseline_sird['I'][baseline_peak_idx]),
                    'error_to_real_day': float(abs(optimized_sird['t'][optimized_peak_idx] - t_sim[data_peak_idx]) - 
                                            abs(baseline_sird['t'][baseline_peak_idx] - t_sim[data_peak_idx]))
                }
            },
            'plots': []
        }
        
        print(f"   ✅ SIRD comparison complete")
        print(f"   Baseline peak: day {comparison_results['peak_analysis']['baseline']['day']:.1f}, "
            f"height {comparison_results['peak_analysis']['baseline']['height']:.0f}")
        print(f"   Optimized peak: day {comparison_results['peak_analysis']['optimized']['day']:.1f}, "
            f"height {comparison_results['peak_analysis']['optimized']['height']:.0f}")
        
        print(f"   ✅ SIRD comparison complete")
    # ============================================================
    # 9.5. Создание сводного отчета
    # ============================================================
    print("\n" + "-" * 60)
    print("📌 STEP 9: Creating summary report")
    print("-" * 60)
    
    import torch  # Добавьте в начало файла если еще нет
    
    report_path = create_summary_report(
        comparison_results=comparison_results,
        history=history,
        baseline_episode=baseline_episode,
        optimized_episode=optimized_episode,
        expert_comment=expert_comment,
        n_epoch=n_epoch,
        lambda_data=lambda_data,
        lambda_ode=lambda_ode,
        lambda_ic=lambda_ic,
        lambda_bc=lambda_bc,
        output_path=None
    )
    
    # ============================================================
    # 10. Итоговый отчет
    # ============================================================
    print("\n" + "=" * 80)
    print("📋 FINAL TEST REPORT")
    print("=" * 80)
    
    print(f"\n📊 OPTIMIZATION SUMMARY:")
    print(f"   Baseline: β={baseline_params['beta']:.4f}, γ={baseline_params['gamma']:.4f}, μ={baseline_params['mu']:.5f}")
    print(f"   Optimized: β={optimized_params['beta']:.4f}, γ={optimized_params['gamma']:.4f}, μ={optimized_params['mu']:.5f}")
    print(f"   Iterations: {len(history)} total, {len(accepted_episodes)} accepted")
    
    print(f"\n📈 SURROGATE RESULTS:")
    print(f"   Baseline peak: {baseline_episode.peak_position:.1f} days, height: {baseline_episode.peak_height:.0f}")
    print(f"   Optimized peak: {optimized_episode.peak_position:.1f} days, height: {optimized_episode.peak_height:.0f}")
    
    print(f"\n🧠 PINN VALIDATION:")
    if comparison_results.get('baseline_pinn', {}).get('success'):
        print(f"   Baseline PINN: ✓ complete")
    if comparison_results.get('optimized_pinn', {}).get('success'):
        print(f"   Optimized PINN: ✓ complete")
    
    print(f"\n📁 OUTPUT FILES:")
    print(f"   - Pipeline graph: pipeline_graph.png")
    print(f"   - PINN agent results: PINN_agent_results/")
    print(f"   - Comparison results: PINN_comparison_results/")
    print(f"   - Summary report: {report_path}")  # ← Добавить
    
    print("\n" + "=" * 80)
    print(f"✅ TEST COMPLETE at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # После строки с comparison_results, добавить:

    # ============================================================
    # 10. СОЗДАНИЕ СВОДНОГО ГРАФИКА
    # ============================================================
    print("\n" + "-" * 60)
    print("📌 STEP 10: Creating full comparison plot")
    print("-" * 60)

    # 1. Реальные данные (уже есть)
    real_data_for_plot = {
        't': np.arange(len(pinn_data['I'])),
        'I': np.array(pinn_data['I']),
        'S': np.array(pinn_data['S']),
        'R': np.array(pinn_data['R']),
        'D': np.array(pinn_data['D'])
    }

    # 2. Базовый прогноз PINN (из comparison_results)
    baseline_pred = {
        't': comparison_results['t_data'],
        'I': np.array(comparison_results['i_predictions']['baseline']),
        'beta': baseline_params['beta'],
        'gamma': baseline_params['gamma'],
        'mu': baseline_params['mu']
    }

    # 3. Синтетические данные (из surrogate_results последнего принятого эпизода)
    surrogate_results = result.get('surrogate_results', {})
    synthetic_data = {
        't': np.array(surrogate_results['t']) if 't' in surrogate_results else np.arange(400),
        'I': np.array(surrogate_results['I']) if 'I' in surrogate_results else np.zeros(400),
        'beta': optimized_params['beta'],  # β от LLM
        'gamma': optimized_params['gamma'],
        'mu': optimized_params['mu']
    }

    # 4. Финальный прогноз PINN (из pinn_verification)
    pinn_verification = result.get('pinn_verification', {})

    # Если нет в result, пробуем загрузить с диска
    if not pinn_verification or not pinn_verification.get('success'):
        print(f"   ⚠️ pinn_verification not in result, trying to load from disk...")
        import glob
        json_files = glob.glob("PINN_verification_results/verification_*.json")
        if json_files:
            latest_file = max(json_files, key=os.path.getctime)
            print(f"   📂 Loading: {latest_file}")
            with open(latest_file, 'r') as f:
                pinn_verification = json.load(f)
            print(f"   ✅ Loaded from disk")

    if pinn_verification.get('success'):
        # Берем предсказания
        t_final = np.array(pinn_verification['predictions']['t'])
        I_final_mean = np.array(pinn_verification['predictions']['I'])
        final_beta = pinn_verification['pinn_recovered_params']['beta']
        final_gamma = pinn_verification['pinn_recovered_params']['gamma']
        final_mu = pinn_verification['pinn_recovered_params']['mu']
        
        # ✅ Берем доверительные интервалы из uncertainty
        uncertainty = pinn_verification.get('uncertainty', {})
        if uncertainty and 'ci_lower_95_I' in uncertainty:
            ci_lower = np.array(uncertainty['ci_lower_95_I'])
            ci_upper = np.array(uncertainty['ci_upper_95_I'])
            print(f"   ✅ Loaded uncertainty: CI width at peak = {np.mean(ci_upper - ci_lower):.1f}")
        else:
            ci_lower = I_final_mean
            ci_upper = I_final_mean
            print(f"   ⚠️ No uncertainty data, using mean only")
        
        final_pred = {
            't': t_final,
            'I': I_final_mean,
            'ci_lower': ci_lower,   # ← ДЛЯ CI
            'ci_upper': ci_upper,   # ← ДЛЯ CI
            'beta': final_beta,
            'gamma': final_gamma,
            'mu': final_mu
        }
        print(f"   ✅ final_pred created with CI: {'ci_lower' in final_pred}")
    else:
        final_pred = synthetic_data
        print(f"   ⚠️ No pinn_verification results, using synthetic data as fallback")

    train_size_points = pinn_verification.get('train_size', 120)
    t_train_split = t_final[train_size_points - 1] if len(t_final) > train_size_points else None
    
    # Создаем график
    full_plot_path = create_comparison_plot(
        real_data=real_data_for_plot,
        baseline_pinn_pred=baseline_pred,
        synthetic_data=synthetic_data,
        final_pinn_pred=final_pred,
        expert_comment=expert_comment,
        train_split_time=int(t_train_split*2.5),
        output_path=None,
        ci_smoothing_sigma=10
    )

    print(f"📊 Full comparison plot: {full_plot_path}")
    
    return result, comparison_results

def create_summary_report(
    comparison_results: dict,
    history: list,
    baseline_episode,
    optimized_episode,
    expert_comment: str,
    n_epoch: int,
    lambda_data: float,
    lambda_ode: float,
    lambda_ic: float,
    lambda_bc: float,
    output_path: str = None
):
    """
    Создает сводный отчет в виде одного изображения
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import matplotlib.gridspec as gridspec
    
    # Создаем фигуру
    fig = plt.figure(figsize=(16, 20))
    gs = gridspec.GridSpec(5, 2, height_ratios=[0.5, 1.5, 1, 1.5, 1], hspace=0.4, wspace=0.3)

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"PINN_comparison_results/summary_report_{timestamp}.png"
    
    # Цвета
    header_color = '#2C3E50'
    text_color = '#2C3E50'
    baseline_color = '#3498DB'
    optimized_color = '#E74C3C'
    accept_color = '#27AE60'
    reject_color = '#E74C3C'
    
    # ============================================================
    # 1. ЗАГОЛОВОК
    # ============================================================
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    title_text = f"PINN VALIDATION REPORT\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    ax_title.text(0.5, 0.7, title_text, fontsize=20, fontweight='bold', 
                  ha='center', va='center', color=header_color)
    ax_title.text(0.5, 0.3, f"Expert comment: '{expert_comment}'", 
                  fontsize=14, ha='center', va='center', 
                  color=text_color, style='italic',
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='#F0F0F0', alpha=0.8))
    
    # ============================================================
    # 2. ПАРАМЕТРЫ ОБУЧЕНИЯ
    # ============================================================
    ax_params = fig.add_subplot(gs[1, 0])
    ax_params.axis('off')
    ax_params.set_title('Training Parameters', fontsize=14, fontweight='bold', pad=15)
    
    params_text = f"""
    ┌─────────────────────────────────┐
    │ Number of epochs:     {n_epoch:>8}       │
    │ λ_data (data loss):   {lambda_data:>8.3f}       │
    │ λ_ode (ODE loss):     {lambda_ode:>8.3f}       │
    │ λ_ic (initial cond):  {lambda_ic:>8.3f}       │
    │ λ_bc (boundary cond): {lambda_bc:>8.3f}       │
    │ Device:               {'CUDA' if torch.cuda.is_available() else 'CPU':>8}       │
    └─────────────────────────────────┘
    """
    ax_params.text(0.1, 0.5, params_text, fontsize=11, family='monospace',
                   va='center', transform=ax_params.transAxes)
    
    # ============================================================
    # 3. СРАВНЕНИЕ ПАРАМЕТРОВ
    # ============================================================
    ax_param_comp = fig.add_subplot(gs[1, 1])
    ax_param_comp.axis('off')
    ax_param_comp.set_title('Parameter Comparison', fontsize=14, fontweight='bold', pad=15)
    
    baseline = comparison_results['baseline_params_input']
    optimized = comparison_results['optimized_params_input']
    baseline_est = comparison_results['baseline_pinn_estimated']
    optimized_est = comparison_results['optimized_pinn_estimated']
    
    param_text = f"""
    ╔═════════╦══════════════╦══════════════╦═══════════════╗
    ║ Param   ║ Baseline     ║ Optimized    ║ Change        ║
    ╠═════════╬══════════════╬══════════════╬═══════════════╣
    ║ β       ║ {baseline['beta']:.4f}        ║ {optimized['beta']:.4f}        ║ {optimized['beta'] - baseline['beta']:+.4f}          ║
    ║ γ       ║ {baseline['gamma']:.4f}        ║ {optimized['gamma']:.4f}        ║ {optimized['gamma'] - baseline['gamma']:+.4f}          ║
    ║ μ       ║ {baseline['mu']:.5f}       ║ {optimized['mu']:.5f}       ║ {optimized['mu'] - baseline['mu']:+.5f}         ║
    ╠═════════╬══════════════╬══════════════╬═══════════════╣
    ║ PINN β  ║ {baseline_est['beta']:.4f}        ║ {optimized_est['beta']:.4f}        ║ {optimized_est['beta'] - baseline_est['beta']:+.4f}          ║
    ║ PINN γ  ║ {baseline_est['gamma']:.4f}        ║ {optimized_est['gamma']:.4f}        ║ {optimized_est['gamma'] - baseline_est['gamma']:+.4f}          ║
    ║ PINN μ  ║ {baseline_est['mu']:.5f}       ║ {optimized_est['mu']:.5f}       ║ {optimized_est['mu'] - baseline_est['mu']:+.5f}         ║
    ╚═════════╩══════════════╩══════════════╩═══════════════╝
    """
    ax_param_comp.text(0.05, 0.5, param_text, fontsize=10, family='monospace',
                       va='center', transform=ax_param_comp.transAxes)
    
    # ============================================================
    # 4. COMPLETE HISTORY (таблица)
    # ============================================================
    ax_history = fig.add_subplot(gs[2, :])
    ax_history.axis('off')
    ax_history.set_title('Optimization History', fontsize=14, fontweight='bold', pad=15)
    
    # Создаем таблицу
    table_data = [['Status', 'Iter', 'β', 'γ', 'μ', 'Peak day', 'Height', 'Deaths']]
    for ep in history:
        status = '✓' if ep.accepted else '✗'
        marker = ' (BL)' if ep.iteration == 0 else ''
        table_data.append([
            status,
            f"{ep.iteration}{marker}",
            f"{ep.beta:.4f}",
            f"{ep.gamma:.4f}",
            f"{ep.mu:.5f}",
            f"{ep.peak_position:.1f}",
            f"{ep.peak_height:.0f}",
            f"{ep.total_deaths:.0f}"
        ])
    
    table = ax_history.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Цвета для строк
    for i, ep in enumerate(history, start=1):
        color = accept_color if ep.accepted else reject_color
        if ep.iteration == 0:
            color = baseline_color
        for j in range(8):
            table[(i, j)].set_facecolor(color)
            table[(i, j)].set_alpha(0.15)
            table[(i, j)].set_text_props(weight='bold' if ep.accepted else 'normal')
    
    # Заголовок таблицы
    for j in range(8):
        table[(0, j)].set_facecolor(header_color)
        table[(0, j)].set_text_props(color='white', weight='bold')
    
    # ============================================================
    # 5. ГРАФИК I(t)
    # ============================================================
    ax_i = fig.add_subplot(gs[3, :])

    # Берем данные из comparison_results
    peak_data = comparison_results['peak_analysis']
    t_data = comparison_results.get('t_data', [])
    I_real = comparison_results.get('I_real', [])
    I_pred_bl = comparison_results.get('i_predictions', {}).get('baseline', [])
    I_pred_opt = comparison_results.get('i_predictions', {}).get('optimized', [])
    train_size = comparison_results.get('train_size')

    baseline_peak_day = peak_data['baseline']['day']
    baseline_peak_height = peak_data['baseline']['height']
    optimized_peak_day = peak_data['optimized']['day']
    optimized_peak_height = peak_data['optimized']['height']
    data_peak_day = peak_data['real_data']['day']
    data_peak_height = peak_data['real_data']['height']

    # Строим кривые
    if len(t_data) > 0:
        ax_i.plot(t_data, I_real, 'o', ms=2, color='gray', alpha=0.4, label='Real data')
        ax_i.plot(t_data, I_pred_bl, '-', linewidth=2, color=baseline_color, alpha=0.8, label='Baseline PINN')
        ax_i.plot(t_data, I_pred_opt, '-', linewidth=2, color=optimized_color, alpha=0.8, label='Optimized PINN')
        ax_i.axvline(train_size, color='gray', linestyle='--', alpha=0.5, label='Train/test split')

    # Отмечаем пики
    ax_i.scatter(baseline_peak_day, baseline_peak_height, color=baseline_color, s=120, 
                marker='v', edgecolors='black', linewidth=1.5, zorder=5, label=f'Baseline peak')
    ax_i.scatter(optimized_peak_day, optimized_peak_height, color=optimized_color, s=120, 
                marker='^', edgecolors='black', linewidth=1.5, zorder=5, label=f'Optimized peak')
    ax_i.scatter(data_peak_day, data_peak_height, color='gray', s=120, 
                marker='s', edgecolors='black', linewidth=1.5, zorder=5, label=f'Real peak')

    # Текст с изменениями
    changes = peak_data['changes']
    ax_i.text(0.98, 0.98, 
            f"Δ day: {changes['peak_day']:+.0f} days\nΔ height: {changes['peak_height']:+.0f} infected",
            fontsize=11, ha='right', va='top', transform=ax_i.transAxes,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='orange', alpha=0.9))

    ax_i.set_xlabel('Days', fontsize=11)
    ax_i.set_ylabel('Infected (I)', fontsize=11)
    ax_i.set_title('I(t) Predictions Comparison', fontsize=14, fontweight='bold', pad=15)
    ax_i.legend(loc='upper left', fontsize=9)
    ax_i.grid(True, alpha=0.3)
    
    # ============================================================
    # 6. ИТОГОВЫЙ ВЫВОД
    # ============================================================
    ax_summary = fig.add_subplot(gs[4, :])
    ax_summary.axis('off')
    
    changes = peak_data['changes']
    day_change = changes['peak_day']
    height_change = changes['peak_height']
    
    # Определяем успешность
    if abs(day_change) > 0 or abs(height_change) > 0:
        success = "✅ Optimization successfully changed peak characteristics"
    else:
        success = "⚠️ No significant changes in peak characteristics"
    
    summary_text = f"""
    ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
    ║                                          SUMMARY                                              ║
    ╠══════════════════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                              ║
    ║   {success:<90}║
    ║                                                                                              ║
    ║   Peak Day Change:     {day_change:+>8.1f} days  │  Peak Height Change:  {height_change:+>8.0f} infected    ║
    ║                                                                                              ║
    ║   Baseline → Optimized:                                                                      ║
    ║   • Peak day:     {peak_data['baseline']['day']:.0f} → {peak_data['optimized']['day']:.0f} days                                       ║
    ║   • Peak height:  {peak_data['baseline']['height']:.0f} → {peak_data['optimized']['height']:.0f} infected                                  ║
    ║                                                                                              ║
    ║   Real Data Reference:                                                                       ║
    ║   • Peak day:     {peak_data['real_data']['day']:.0f} days                                                              ║
    ║   • Peak height:  {peak_data['real_data']['height']:.0f} infected                                                         ║
    ║                                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
    """
    
    ax_summary.text(0.05, 0.5, summary_text, fontsize=11, family='monospace',
                    va='center', transform=ax_summary.transAxes)
    
    # ============================================================
    # Сохранение
    # ============================================================
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n📄 Summary report saved: {output_path}")
    return output_path

def create_comparison_plot(
    real_data: dict,           # {'t': array, 'I': array, 'S': array, 'R': array, 'D': array}
    baseline_pinn_pred: dict,  # {'t': array, 'I': array, 'beta': float, 'gamma': float, 'mu': float}
    synthetic_data: dict,      # {'t': array, 'I': array, 'beta': float, 'gamma': float, 'mu': float}
    final_pinn_pred: dict,     # {'t': array, 'I': array, 'ci_lower': array, 'ci_upper': array, 'beta': float, 'gamma': float, 'mu': float}
    expert_comment: str,
    train_split_time: float = None, 
    output_path: str = None,
    ci_smoothing_sigma: float = 2.0
):
    """
    Создает сравнительный график всех кривых I(t) с доверительным интервалом для финального PINN.
    Сохраняет PNG и отдельные PDF графики.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from datetime import datetime
    import os
    from scipy.ndimage import gaussian_filter1d
    
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"PINN_comparison_results/full_comparison_{timestamp}.png"
    
    os.makedirs("PINN_comparison_results", exist_ok=True)
    
    # Устанавливаем стиль для более приятного вида
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Full Comparison: Real Data → Baseline PINN → Synthetic data (SIRD) → Final PINN\nExpert: "{expert_comment[:80]}"', 
                fontsize=12, fontweight='bold')
    
    # ============================================================
    # 1. I(t) - основной график
    # ============================================================
    ax1 = axes[0, 0]
    
    # Реальные данные - ДО train_size: черные точки, ПОСЛЕ: белые с черной границей
    t_real = real_data['t']
    I_real = real_data['I']
    
    # Данные ДО train_size (черные)
    if train_split_time is not None:
        train_mask = t_real <= train_split_time
        ax1.plot(t_real[train_mask][::7], I_real[train_mask][::7], 'o', ms=5, color='black', alpha=0.8, 
                 label='Real data', markeredgecolor='black', markeredgewidth=0.5, zorder=10)
        
        # Данные ПОСЛЕ train_size (белые с черной границей)
        test_mask = t_real > train_split_time
        ax1.scatter(t_real[test_mask][::7], I_real[test_mask][::7], s=25, facecolor='white', 
                    edgecolor='black', linewidth=1.2, alpha=1.0, 
                    label='Future data', zorder=10)
    else:
        ax1.plot(t_real[::7], I_real[::7], 'o', ms=5, color='black', alpha=0.8, 
                 label='Real data', markeredgecolor='black', markeredgewidth=0.5, zorder=10)
    
    # Синтетические данные - ОРАНЖЕВАЯ ПУНКТИРНАЯ ЛИНИЯ
    t_synth = synthetic_data['t']
    I_synth = synthetic_data['I']
    synth_beta = synthetic_data.get('beta', 0)
    ax1.plot(t_synth, I_synth, '--', linewidth=2.5, color='#FF8C00', alpha=0.9, 
             label=f'Synthetic data (LLM β={synth_beta:.4f})', zorder=15)
    
    # Начальный прогноз - КРАСНАЯ ЛИНИЯ
    t_base = baseline_pinn_pred['t']
    I_base = baseline_pinn_pred['I']
    ax1.plot(t_base, I_base, '-', linewidth=2, color='#D32F2F', alpha=0.8, 
             label=f'Initial forecast (β={baseline_pinn_pred.get("beta", 0):.4f})', zorder=15)
    
    # Финальный прогноз - СИНЯЯ ЛИНИЯ
    t_final = final_pinn_pred['t']
    I_final = final_pinn_pred['I']
    final_beta = final_pinn_pred.get('beta', 0)
    
    ax1.plot(t_final, I_final, '-', linewidth=2.5, color='#1565C0', alpha=0.9, 
             label=f'Final forecast (β={final_beta:.4f})', zorder=8)
    
    # Доверительный интервал (95% CI)
    ci_lower_smooth = None
    ci_upper_smooth = None
    
    if 'ci_lower' in final_pinn_pred and 'ci_upper' in final_pinn_pred:
        ci_lower = np.array(final_pinn_pred['ci_lower'])
        ci_upper = np.array(final_pinn_pred['ci_upper'])
        
        # Полупрозрачная область CI
        ax1.fill_between(t_final, ci_lower, ci_upper, 
                         alpha=0.25, color='#64B5F6', label='95% CI', zorder=4)
        
        # Gaussian filter для сглаживания
        ci_lower_smooth = gaussian_filter1d(ci_lower, sigma=ci_smoothing_sigma)
        ci_upper_smooth = gaussian_filter1d(ci_upper, sigma=ci_smoothing_sigma)
        
        # Сглаженные границы CI
        ax1.plot(t_final, ci_lower_smooth, '-', linewidth=1.2, color='#42A5F5', alpha=0.6, zorder=6)
        ax1.plot(t_final, ci_upper_smooth, '-', linewidth=1.2, color='#42A5F5', alpha=0.6, zorder=6)
    
    # Линия train/test split
    if train_split_time is not None:
        ax1.axvline(train_split_time, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, 
                   label='Train/Test split', zorder=3)
    
    ax1.set_xlabel('Days', fontsize=11)
    ax1.set_ylabel('Infected (I)', fontsize=11)
    ax1.set_title('I(t) Comparison', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=7, ncol=1, framealpha=0.9, fancybox=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # ============================================================
    # 2. I(t) - логарифмический масштаб
    # ============================================================
    ax2 = axes[0, 1]
    
    if train_split_time is not None:
        train_mask = t_real <= train_split_time
        ax2.semilogy(t_real[train_mask][::7], I_real[train_mask][::7] + 1, 'o', ms=5, 
                     color='black', alpha=0.8, label='Real data (train)', 
                     markeredgecolor='black', markeredgewidth=0.5, zorder=10)
        
        test_mask = t_real > train_split_time
        ax2.scatter(t_real[test_mask][::7], I_real[test_mask][::7] + 1, s=25, 
                    facecolor='white', edgecolor='black', linewidth=1.2, alpha=1.0, 
                    label='Real data (test)', zorder=10)
    else:
        ax2.semilogy(t_real[::7], I_real[::7] + 1, 'o', ms=5, color='black', alpha=0.8, 
                     label='Real data', zorder=10)
    
    ax2.semilogy(t_synth, I_synth + 1, '--', linewidth=2.5, color='#FF8C00', alpha=0.9, 
                 label='Synthetic data (LLM β)', zorder=15)
    ax2.semilogy(t_base, I_base + 1, '-', linewidth=2, color='#D32F2F', alpha=0.8, 
                 label='Initial forecast', zorder=5)
    ax2.semilogy(t_final, I_final + 1, '-', linewidth=2.5, color='#1565C0', alpha=0.9, 
                 label='Final forecast', zorder=8)
    
    if ci_lower_smooth is not None:
        ax2.fill_between(t_final, ci_lower + 1, ci_upper + 1, 
                         alpha=0.25, color='#64B5F6', label='95% CI', zorder=4)
        ax2.semilogy(t_final, ci_lower_smooth + 1, '-', linewidth=1.2, color='#42A5F5', alpha=0.6, zorder=6)
        ax2.semilogy(t_final, ci_upper_smooth + 1, '-', linewidth=1.2, color='#42A5F5', alpha=0.6, zorder=6)
    
    if train_split_time is not None:
        ax2.axvline(train_split_time, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax2.set_xlabel('Days', fontsize=11)
    ax2.set_ylabel('Infected (I) - log scale', fontsize=11)
    ax2.set_title('I(t) Comparison (Log Scale)', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=7, framealpha=0.9, fancybox=True)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # ============================================================
    # 3. Сравнение параметров
    # ============================================================
    ax3 = axes[1, 0]
    
    params_names = ['β', 'γ', 'μ']
    baseline_values = [
        baseline_pinn_pred.get('beta', 0),
        baseline_pinn_pred.get('gamma', 0),
        baseline_pinn_pred.get('mu', 0)
    ]
    synth_values = [
        synthetic_data.get('beta', 0),
        synthetic_data.get('gamma', 0),
        synthetic_data.get('mu', 0)
    ]
    final_values = [
        final_pinn_pred.get('beta', 0),
        final_pinn_pred.get('gamma', 0),
        final_pinn_pred.get('mu', 0)
    ]
    
    x = np.arange(len(params_names))
    width = 0.25
    
    bars1 = ax3.bar(x - width, baseline_values, width, label='Initial PINN', 
                    color='#D32F2F', alpha=0.8, edgecolor='#B71C1C', linewidth=1)
    bars2 = ax3.bar(x, synth_values, width, label='Synthetic SIRD (LLM β)', 
                    color='#FF8C00', alpha=0.8, edgecolor='#CC7000', linewidth=1)
    bars3 = ax3.bar(x + width, final_values, width, label='Final PINN', 
                    color='#1565C0', alpha=0.8, edgecolor='#0D47A1', linewidth=1)
    
    ax3.set_ylabel('Parameter value', fontsize=11)
    ax3.set_title('Parameter Comparison', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(params_names, fontsize=11)
    ax3.legend(fontsize=8, loc='upper left', framealpha=0.9, fancybox=True)
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    for bar, val in zip(bars1, baseline_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
                f'{val:.4f}', ha='center', va='bottom', fontsize=8, rotation=45)
    for bar, val in zip(bars2, synth_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
                f'{val:.4f}', ha='center', va='bottom', fontsize=8, rotation=45)
    for bar, val in zip(bars3, final_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
                f'{val:.4f}', ha='center', va='bottom', fontsize=8, rotation=45)
    
    # ============================================================
    # 4. Peak analysis
    # ============================================================
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    real_peak_idx = np.argmax(I_real)
    base_peak_idx = np.argmax(I_base)
    synth_peak_idx = np.argmax(I_synth)
    final_peak_idx = np.argmax(I_final)
    
    real_peak_day = t_real[real_peak_idx] if len(t_real) > real_peak_idx else 0
    real_peak_height = I_real[real_peak_idx]
    base_peak_day = t_base[base_peak_idx]
    base_peak_height = I_base[base_peak_idx]
    synth_peak_day = t_synth[synth_peak_idx]
    synth_peak_height = I_synth[synth_peak_idx]
    final_peak_day = t_final[final_peak_idx]
    final_peak_height = I_final[final_peak_idx]
    
    ci_width_at_peak = ""
    if ci_lower_smooth is not None:
        ci_width = ci_upper_smooth[final_peak_idx] - ci_lower_smooth[final_peak_idx]
        ci_width_at_peak = f"\n   95% CI width at peak: {ci_width:.0f}"
    
    text = f"""
┌────────────────────────────────────────────────────────────────┐
│                        PEAK ANALYSIS                           │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   {'':<20} {'Day':<12} {'Height':<15}                         │
│   {'─'*50}                                                     │
│   {'Real data':<20} {real_peak_day:<12.1f} {real_peak_height:<15.0f}      │
│   {'Initial PINN':<20} {base_peak_day:<12.1f} {base_peak_height:<15.0f}      │
│   {'Synthetic SIRD':<20} {synth_peak_day:<12.1f} {synth_peak_height:<15.0f}      │
│   {'Final PINN':<20} {final_peak_day:<12.1f} {final_peak_height:<15.0f}      │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│                          CHANGES                               │
├────────────────────────────────────────────────────────────────┤
│   Initial → Synthetic: Δβ = {synth_values[0] - baseline_values[0]:+.4f}                            │
│   Initial → Final:     Δβ = {final_values[0] - baseline_values[0]:+.4f}                            │
│   Synthetic → Final:   Δβ = {final_values[0] - synth_values[0]:+.4f}                            │
│                                                                │
│   Peak day change (Initial → Final): {final_peak_day - base_peak_day:+.1f} days                    │
│   Peak height change (Initial → Final): {final_peak_height - base_peak_height:+.0f} infected{ci_width_at_peak}
│                                                                │
└────────────────────────────────────────────────────────────────┘
"""
    
    ax4.text(0.05, 0.95, text, fontsize=9, family='monospace', va='top', transform=ax4.transAxes)
    
    # ============================================================
    # Сохранение отдельных графиков в PDF
    # ============================================================
    pdf_dir = "PINN_comparison_results/pdf_plots"
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Основной график I(t)
    fig_i_main = plt.figure(figsize=(10, 7))
    ax_main = fig_i_main.add_subplot(111)
    
    
    
    
    ax_main.plot(t_base, I_base, '-', linewidth=2, color='#D32F2F', alpha=0.8,
                 label=f'Initial forecast (β={baseline_pinn_pred.get("beta", 0):.4f})')
    ax_main.plot(t_final, I_final, '-', linewidth=2.5, color='#1565C0', alpha=0.9,
                 label=f'Final forecast (β={final_beta:.4f})')
    ax_main.plot(t_synth, I_synth, '--', linewidth=2.5, color='#FF8C00', alpha=0.9,
                 label=f'Synthetic data (LLM β={synth_beta:.4f})')
    if train_split_time is not None:
        train_mask = t_real <= train_split_time
        ax_main.plot(t_real[train_mask][::7], I_real[train_mask][::7], 'o', ms=5, color='black', alpha=0.8,
                     label='Real data', markeredgecolor='black', markeredgewidth=0.5)
        test_mask = t_real > train_split_time
        ax_main.scatter(t_real[test_mask][::7], I_real[test_mask][::7], s=25, facecolor='white',
                edgecolor='black', linewidth=1.2, alpha=1.0, label='Future data', zorder=20)
    else:
        ax_main.plot(t_real[::7], I_real[::7], 'o', ms=5, color='black', alpha=0.8,
                     label='Real data', markeredgecolor='black', markeredgewidth=0.5)
    
    if ci_lower_smooth is not None:
        ax_main.fill_between(t_final, ci_lower, ci_upper, alpha=0.25, color='#64B5F6', label='95% CI')
        ax_main.plot(t_final, ci_lower_smooth, '-', linewidth=1.2, color='#42A5F5', alpha=0.6)
        ax_main.plot(t_final, ci_upper_smooth, '-', linewidth=1.2, color='#42A5F5', alpha=0.6)
    
    if train_split_time is not None:
        ax_main.axvline(train_split_time, color='gray', linestyle='--', linewidth=1.5, alpha=0.7,
                        label='Train/Test split')
    
    ax_main.set_xlabel('Days', fontsize=12)
    ax_main.set_ylabel('Infected (I)', fontsize=12)
    # ax_main.set_title(f'I(t) Comparison\nExpert: "{expert_comment[:60]}"', fontsize=14, fontweight='bold')
    ax_main.legend(loc='upper left', fontsize=9, framealpha=0.9, fancybox=True)
    ax_main.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    pdf_path = os.path.join(pdf_dir, f"I_plot_{pdf_timestamp}.pdf")
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    print(f"   ✅ PDF saved: {pdf_path}")
    
    
    
    # ============================================================
    # Сохранение основного PNG
    # ============================================================
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Full comparison plot saved: {output_path}")
    return output_path

if __name__ == "__main__":
    result, comparison = main()