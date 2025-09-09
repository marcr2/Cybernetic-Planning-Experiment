"""
Cybernetic Feedback Implementation

Implements cybernetic feedback mechanisms for the economic planning system,
including circular causality, self - regulation, and adaptive control.

Based on Stafford Beer's cybernetic principles and the Viable System Model (VSM).
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np

class CyberneticFeedbackSystem:
    """
    Implements cybernetic feedback mechanisms for economic planning.

    Based on cybernetic principles:
    - Circular causality: outputs influence inputs - Self - regulation: system maintains equilibrium - Requisite variety: control system has sufficient complexity - Feedback loops: information flows back to control mechanisms
    """

    def __init__(
        self,
        technology_matrix: np.ndarray,
        final_demand: np.ndarray,
        labor_vector: np.ndarray,
        feedback_strength: float = 0.1,
        adaptation_rate: float = 0.05,
        convergence_threshold: float = 1e-6,
        max_iterations: int = 100
    ):
        """
        Initialize the cybernetic feedback system.

        Args:
            technology_matrix: Technology matrix A
            final_demand: Final demand vector d
            labor_vector: Labor input vector l
            feedback_strength: Strength of feedback loops (0 - 1)
            adaptation_rate: Rate of system adaptation (0 - 1)
            convergence_threshold: Convergence tolerance
            max_iterations: Maximum iterations for convergence
        """
        self.A = np.asarray(technology_matrix)
        self.d = np.asarray(final_demand).flatten()
        self.l = np.asarray(labor_vector).flatten()
        self.n_sectors = self.A.shape[0]

        # Cybernetic parameters
        self.feedback_strength = feedback_strength
        self.adaptation_rate = adaptation_rate
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations

        # State variables
        self.current_output = None
        self.current_demand = self.d.copy()
        self.iteration = 0
        self.converged = False

        # Feedback history for analysis
        self.feedback_history = []
        self.output_history = []
        self.demand_history = []

    def apply_cybernetic_feedback(self, initial_output: np.ndarray) -> Dict[str, Any]:
        """
        Apply cybernetic feedback to the economic system.

        Implements the cybernetic equation:
        x_{t + 1} = (I - A)^{-1} * (d_t + feedback_adjustment)

        Args:
            initial_output: Initial output vector

        Returns:
            Dictionary with feedback results
        """
        self.current_output = initial_output.copy()
        self.current_demand = self.d.copy()

        # Calculate Leontief inverse
        I = np.eye(self.n_sectors)
        try:
            leontief_inverse = np.linalg.inv(I - self.A)
        except np.linalg.LinAlgError:
            raise ValueError("Cannot compute Leontief inverse - economy not productive")

        # Apply cybernetic feedback loop
        for iteration in range(self.max_iterations):
            self.iteration = iteration

            # Store current state
            self.output_history.append(self.current_output.copy())
            self.demand_history.append(self.current_demand.copy())

            # Calculate feedback adjustment based on output - demand mismatch
            output_demand = self.A @ self.current_output
            net_output = self.current_output - output_demand
            demand_satisfaction = net_output / (self.current_demand + 1e-10)  # Avoid division by zero

            # Calculate feedback signal
            feedback_signal = self._calculate_feedback_signal(demand_satisfaction)

            # Apply feedback to demand
            demand_adjustment = self.feedback_strength * feedback_signal
            self.current_demand = self.d + demand_adjustment

            # Ensure non - negative demand
            self.current_demand = np.maximum(self.current_demand, 0)

            # Calculate new output using Leontief model
            new_output = leontief_inverse @ self.current_demand

            # Check for convergence
            output_change = np.linalg.norm(new_output - self.current_output)
            if output_change < self.convergence_threshold:
                self.converged = True
                break

            # Update output with adaptation
            self.current_output = (1 - self.adaptation_rate) * self.current_output + \
                                self.adaptation_rate * new_output

            # Store feedback information
            self.feedback_history.append({
                'iteration': iteration,
                'output_change': output_change,
                'feedback_signal': feedback_signal.copy(),
                'demand_adjustment': demand_adjustment.copy(),
                'convergence_ratio': output_change / (np.linalg.norm(self.current_output) + 1e-10)
            })

        # Calculate final results
        final_net_output = self.current_output - (self.A @ self.current_output)
        final_demand_satisfaction = final_net_output / (self.current_demand + 1e-10)

        return {
            'final_output': self.current_output,
            'final_demand': self.current_demand,
            'final_net_output': final_net_output,
            'demand_satisfaction': final_demand_satisfaction,
            'converged': self.converged,
            'iterations': self.iteration + 1,
            'feedback_history': self.feedback_history,
            'output_history': self.output_history,
            'demand_history': self.demand_history,
            'cybernetic_metrics': self._calculate_cybernetic_metrics()
        }

    def _calculate_feedback_signal(self, demand_satisfaction: np.ndarray) -> np.ndarray:
        """
        Calculate cybernetic feedback signal.

        Implements proportional - integral - derivative (PID) control:
        - Proportional: immediate response to error - Integral: accumulated error over time - Derivative: rate of change of error

        Args:
            demand_satisfaction: Current demand satisfaction ratios

        Returns:
            Feedback signal vector
        """
        # Proportional term: immediate response to demand satisfaction
        proportional = 1.0 - demand_satisfaction

        # Integral term: accumulated demand satisfaction over iterations
        if len(self.feedback_history) > 0:
            # Use exponential moving average for integral term
            alpha = 0.3
            if hasattr(self, '_integral_term'):
                self._integral_term = alpha * proportional + (1 - alpha) * self._integral_term
            else:
                self._integral_term = proportional
        else:
            self._integral_term = proportional

        # Derivative term: rate of change of demand satisfaction
        if len(self.feedback_history) > 1:
            prev_satisfaction = self.feedback_history[-1].get('demand_satisfaction', demand_satisfaction)
            derivative = demand_satisfaction - prev_satisfaction
        else:
            derivative = np.zeros_like(demand_satisfaction)

        # PID controller parameters
        kp = 1.0  # Proportional gain
        ki = 0.5  # Integral gain
        kd = 0.1  # Derivative gain

        # Calculate PID output
        feedback_signal = kp * proportional + ki * self._integral_term + kd * derivative

        # Apply cybernetic constraints
        feedback_signal = self._apply_cybernetic_constraints(feedback_signal)

        return feedback_signal

    def _apply_cybernetic_constraints(self, feedback_signal: np.ndarray) -> np.ndarray:
        """
        Apply cybernetic constraints to feedback signal.

        Ensures the feedback system maintains:
        - Stability: bounded responses - Causality: future doesn't affect past - Physical realizability: non - negative outputs

        Args:
            feedback_signal: Raw feedback signal

        Returns:
            Constrained feedback signal
        """
        # Stability constraint: limit maximum feedback
        max_feedback = 0.5 * self.d  # Maximum 50% of original demand
        feedback_signal = np.clip(feedback_signal, -max_feedback, max_feedback)

        # Causality constraint: ensure non - negative total demand
        total_demand = self.d + feedback_signal
        if np.any(total_demand < 0):
            # Scale down feedback to maintain non - negative demand
            scale_factor = np.min(self.d / (self.d - feedback_signal + 1e-10))
            scale_factor = min(scale_factor, 1.0)
            feedback_signal *= scale_factor

        # Requisite variety constraint: limit complexity of feedback
        # Apply smoothing to prevent oscillatory behavior
        if len(self.feedback_history) > 2:
            # Use exponential smoothing
            alpha = 0.7
            prev_feedback = self.feedback_history[-1]['feedback_signal']
            feedback_signal = alpha * feedback_signal + (1 - alpha) * prev_feedback

        return feedback_signal

    def _calculate_cybernetic_metrics(self) -> Dict[str, Any]:
        """
        Calculate cybernetic performance metrics.

        Returns:
            Dictionary with cybernetic metrics
        """
        if not self.feedback_history:
            return {}

        # Stability metrics
        output_changes = [h['output_change'] for h in self.feedback_history]
        stability = 1.0 / (1.0 + np.std(output_changes))  # Higher is more stable

        # Responsiveness metrics
        convergence_rate = 1.0 / (self.iteration + 1) if self.iteration > 0 else 0

        # Efficiency metrics
        final_demand_satisfaction = self.feedback_history[-1].get('demand_satisfaction', np.array([0]))
        efficiency = np.mean(final_demand_satisfaction)

        # Adaptability metrics
        feedback_variance = np.var([h['feedback_signal'] for h in self.feedback_history])
        adaptability = 1.0 / (1.0 + feedback_variance)  # Higher is more adaptable

        return {
            'stability': stability,
            'convergence_rate': convergence_rate,
            'efficiency': efficiency,
            'adaptability': adaptability,
            'total_iterations': self.iteration + 1,
            'converged': self.converged,
            'final_output_change': output_changes[-1] if output_changes else 0,
            'cybernetic_health': (stability + efficiency + adaptability) / 3
        }

    def reset_feedback_state(self):
        """Reset the feedback system state."""
        self.current_output = None
        self.current_demand = self.d.copy()
        self.iteration = 0
        self.converged = False
        self.feedback_history = []
        self.output_history = []
        self.demand_history = []
        if hasattr(self, '_integral_term'):
            delattr(self, '_integral_term')

    def update_cybernetic_parameters(
        self,
        feedback_strength: Optional[float] = None,
        adaptation_rate: Optional[float] = None,
        convergence_threshold: Optional[float] = None
    ):
        """
        Update cybernetic parameters dynamically.

        Args:
            feedback_strength: New feedback strength
            adaptation_rate: New adaptation rate
            convergence_threshold: New convergence threshold
        """
        if feedback_strength is not None:
            self.feedback_strength = np.clip(feedback_strength, 0.0, 1.0)
        if adaptation_rate is not None:
            self.adaptation_rate = np.clip(adaptation_rate, 0.0, 1.0)
        if convergence_threshold is not None:
            self.convergence_threshold = max(convergence_threshold, 1e-10)

    def get_system_diagnostics(self) -> Dict[str, Any]:
        """
        Get comprehensive system diagnostics.

        Returns:
            Dictionary with system diagnostics
        """
        diagnostics = {
            'cybernetic_parameters': {
                'feedback_strength': self.feedback_strength,
                'adaptation_rate': self.adaptation_rate,
                'convergence_threshold': self.convergence_threshold,
                'max_iterations': self.max_iterations
            },
            'current_state': {
                'output': self.current_output.tolist() if self.current_output is not None else None,
                'demand': self.current_demand.tolist(),
                'iteration': self.iteration,
                'converged': self.converged
            },
            'system_metrics': self._calculate_cybernetic_metrics(),
            'feedback_analysis': self._analyze_feedback_patterns()
        }

        return diagnostics

    def _analyze_feedback_patterns(self) -> Dict[str, Any]:
        """
        Analyze feedback patterns for cybernetic insights.

        Returns:
            Dictionary with feedback pattern analysis
        """
        if len(self.feedback_history) < 2:
            return {'error': 'Insufficient feedback history for analysis'}

        # Analyze convergence pattern
        output_changes = [h['output_change'] for h in self.feedback_history]
        convergence_pattern = 'exponential' if len(output_changes) > 2 else 'linear'

        # Analyze feedback signal patterns
        feedback_signals = [h['feedback_signal'] for h in self.feedback_history]
        feedback_variance = np.var(feedback_signals, axis = 0)
        dominant_feedback_sectors = np.argsort(feedback_variance)[-3:]  # Top 3 sectors

        # Analyze stability
        is_stable = all(change < self.convergence_threshold * 10 for change in output_changes[-3:])

        return {
            'convergence_pattern': convergence_pattern,
            'dominant_feedback_sectors': dominant_feedback_sectors.tolist(),
            'feedback_variance': feedback_variance.tolist(),
            'is_stable': is_stable,
            'oscillation_detected': len(set(np.sign(feedback_signals[-1]))) > 1
        }
