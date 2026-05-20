import numpy as np
from typing import Dict, Any
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings('ignore')

class SIRDSurrogate:
    """
    SIRD модель как суррогат для быстрого прогнозирования
    """
    
    def __init__(self, population, S0, I0, R0, D0):
        """
        Args:
            population: общая численность населения
            S0: начальное количество восприимчивых
            I0: начальное количество инфицированных
            R0: начальное количество выздоровевших
            D0: начальное количество умерших
        """
        self.population = population
        self.S0 = S0
        self.I0 = I0
        self.R0 = R0
        self.D0 = D0
        
    def ode_system(self, t, y, beta, gamma, mu):
        S, I, R, D = y
        N = self.population
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I - mu * I
        dRdt = gamma * I
        dDdt = mu * I
        return [dSdt, dIdt, dRdt, dDdt]
    
    
    def simulate(self, beta: float, gamma: float, mu: float, 
                 t_max: int = None, num_points: int = 1000) -> Dict[str, Any]:
        """
        Симуляция SIRD модели
        
        Args:
            beta: скорость заражения
            gamma: скорость выздоровления
            mu: смертность
            t_max: максимальное время симуляции (дни)
            num_points: количество точек для вывода
            
        Returns:
            dict: метрики эпидемии
        """
        t_span = (0, t_max)
        t_eval = np.linspace(0, t_max, num_points)
        y0 = [self.S0, self.I0, self.R0, self.D0]
        
        try:
            # Быстрое численное решение
            solution = solve_ivp(
                self.ode_system, 
                t_span, 
                y0, 
                args=(beta, gamma, mu),
                t_eval=t_eval,
                method='RK45',
                rtol=1e-6,
                atol=1e-8
            )
            
            t = solution.t
            S, I, R, D = solution.y
            
            # Находим пик инфекции
            peak_idx = np.argmax(I)
            peak_position = t[peak_idx]  # день пика
            peak_height = I[peak_idx]     # количество инфицированных в пике
            
            # Дополнительные метрики
            final_infected = I[-1]
            total_recovered = R[-1]
            total_deaths = D[-1]
            
            return {
                'peak_position': peak_position,
                'peak_height': peak_height,
                'peak_day': int(peak_position),
                't': t,
                'I': I,
                'S': S,
                'R': R,
                'D': D,
                'final_infected': final_infected,
                'total_recovered': total_recovered,
                'total_deaths': total_deaths
            }
            
        except Exception as e:
            print(f"Ошибка симуляции для β={beta:.3f}, γ={gamma:.3f}, μ={mu:.5f}: {e}")
            return None


# ============================================
# Surrogate Agent
# ============================================

class SurrogateAgent:
    """
    Агент для суррогатной модели SIRD.
    Начальные условия берутся из state при каждом вызове.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Инициализация суррогатного агента
        
        Args:
            verbose: флаг подробного вывода
        """
        self.verbose = verbose
        self.cache = {}  # Кэш для быстрого доступа
    
    def simulate(self, beta: float, gamma: float, mu: float, 
             population: int, S0: int, I0: int, R0: int, D0: int,
             t_max: int = 200, num_points: int = 1000) -> Dict[str, Any]:
        """
        Симуляция SIRD модели
        """
        # Создаем суррогат с переданными начальными условиями
        surrogate = SIRDSurrogate(
            population=population,
            S0=S0,
            I0=I0,
            R0=R0,
            D0=D0
        )
        
        result = surrogate.simulate(
            beta=beta,
            gamma=gamma,
            mu=mu,
            t_max=t_max,
            num_points=num_points
        )
        
        if result is None:
            return {
                'success': False,
                'error': 'Симуляция не удалась',
                'beta': beta,
                'gamma': gamma,
                'mu': mu
            }
        
        # ✅ Конвертируем numpy типы в стандартные Python типы
        return {
            'success': True,
            'beta': float(beta),
            'gamma': float(gamma),
            'mu': float(mu),
            'peak_position': float(result['peak_position']),
            'peak_height': float(result['peak_height']),
            'peak_day': int(result['peak_day']),
            'total_recovered': float(result['total_recovered']),
            'total_deaths': float(result['total_deaths']),
            'final_infected': float(result['final_infected']),
            't': result['t'].tolist() if hasattr(result['t'], 'tolist') else list(result['t']),
            'I': result['I'].tolist() if hasattr(result['I'], 'tolist') else list(result['I']),
            'S': result['S'].tolist() if hasattr(result['S'], 'tolist') else list(result['S']),
            'R': result['R'].tolist() if hasattr(result['R'], 'tolist') else list(result['R']),
            'D': result['D'].tolist() if hasattr(result['D'], 'tolist') else list(result['D']),
        }
    
    def __call__(self, state: Dict) -> Dict:
        """
        Вызов агента для LangGraph pipeline
        
        Args:
            state: текущее состояние графа, должно содержать:
                - generated_params: dict с ключами beta, gamma, mu
                - initial_conditions: dict с ключами population, S0, I0, R0, D0
                - simulation_params: dict с ключами t_max, num_points (опционально)
                
        Returns:
            обновленное состояние с результатами симуляции
        """
        if self.verbose:
            print("=" * 60)
            print("📊 SURROGATE AGENT")
            print("=" * 60)
        
        # Получаем сгенерированные параметры
        generated_params = state.get('generated_params', {})
        
        if not generated_params:
            if self.verbose:
                print("❌ Нет параметров для симуляции")
            state['surrogate_results'] = {
                'success': False,
                'error': 'Нет параметров для симуляции'
            }
            return state
        
        # Получаем начальные условия из state
        initial_conditions = state.get('initial_conditions', {})
        
        if not initial_conditions:
            if self.verbose:
                print("❌ Нет начальных условий в state")
            state['surrogate_results'] = {
                'success': False,
                'error': 'Нет начальных условий. Укажите population, S0, I0, R0, D0 в state["initial_conditions"]'
            }
            return state
        
        # Извлекаем параметры
        beta = generated_params.get('beta')
        gamma = generated_params.get('gamma')
        mu = generated_params.get('mu')
        
        population = initial_conditions.get('population')
        S0 = initial_conditions.get('S0')
        I0 = initial_conditions.get('I0')
        R0 = initial_conditions.get('R0')
        D0 = initial_conditions.get('D0')
        
        # Проверяем обязательные параметры
        if beta is None or gamma is None or mu is None:
            if self.verbose:
                print(f"❌ Неполные параметры: beta={beta}, gamma={gamma}, mu={mu}")
            state['surrogate_results'] = {
                'success': False,
                'error': f'Неполные параметры: beta={beta}, gamma={gamma}, mu={mu}'
            }
            return state
        
        if population is None or S0 is None or I0 is None or R0 is None or D0 is None:
            if self.verbose:
                print(f"❌ Неполные начальные условия: population={population}, S0={S0}, I0={I0}, R0={R0}, D0={D0}")
            state['surrogate_results'] = {
                'success': False,
                'error': f'Неполные начальные условия'
            }
            return state
        
        # Параметры симуляции (с значениями по умолчанию)
        sim_params = state.get('simulation_params', {})
        t_max = sim_params.get('t_max', 400)
        num_points = sim_params.get('num_points', 1000)
        
        if self.verbose:
            print(f"🚀 Симуляция с параметрами:")
            print(f"   β (infection rate): {beta:.4f}")
            print(f"   γ (recovery rate): {gamma:.4f}")
            print(f"   μ (mortality rate): {mu:.5f}")
            if gamma + mu > 0:
                print(f"   R0 (basic reproduction): {beta / (gamma + mu):.2f}")
            print(f"\n🏥 Начальные условия:")
            print(f"   Population: {population:,}")
            print(f"   S0: {S0:,}, I0: {I0:,}, R0: {R0:,}, D0: {D0:,}")
        
        # Запускаем симуляцию
        results = self.simulate(
            beta=beta, gamma=gamma, mu=mu,
            population=population, S0=S0, I0=I0, R0=R0, D0=D0,
            t_max=t_max, num_points=num_points
        )
        
        if results.get('success', False):
            if self.verbose:
                print(f"✅ Симуляция завершена")
                print(f"   📍 Пик инфекции:")
                print(f"      День: {results['peak_day']:.0f}")
                print(f"      Инфицированных: {results['peak_height']:.0f}")
                print(f"   💀 Итоговая статистика:")
                print(f"      Выздоровевшие: {results['total_recovered']:.0f}")
                print(f"      Умершие: {results['total_deaths']:.0f}")
                print(f"      Осталось инфицированных: {results['final_infected']:.0f}")
        else:
            if self.verbose:
                print(f"❌ Ошибка симуляции: {results.get('error', 'Unknown error')}")
        
        # Сохраняем результаты в состояние
        state['surrogate_results'] = results
        
        # Если есть целевой пик в конфигурации, добавляем оценку
        task_config = state.get('task_config', {})
        target_peak = task_config.get('target_peak')
        
        if target_peak is not None and results.get('success', False):
            peak_error = abs(results['peak_position'] - target_peak)
            state['peak_error'] = peak_error
            state['is_acceptable'] = peak_error <= task_config.get('peak_tolerance', 5.0)
            
            if self.verbose:
                print(f"\n🎯 Оценка относительно целевого пика (день {target_peak}):")
                print(f"   Ошибка: {peak_error:.1f} дней")
                print(f"   Приемлемо: {'✅ Да' if state['is_acceptable'] else '❌ Нет'}")
        
        return state