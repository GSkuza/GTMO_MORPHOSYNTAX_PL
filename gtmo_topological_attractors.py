#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GTMØ Topological Attractors & Temporal Evolution Module
=======================================================
Analyzes semantic attractors, trajectories, and temporal evolution in D-S-E space.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# TOPOLOGICAL ATTRACTORS IN D-S-E SPACE
# =============================================================================

@dataclass
class Attractor:
    """Represents a topological attractor in D-S-E space."""
    name: str
    coordinates: np.ndarray  # [D, S, E]
    basin_radius: float
    stability_index: float
    type: str  # 'fixed_point', 'limit_cycle', 'strange_attractor'


# Define known attractors in GTMØ semantic space
SEMANTIC_ATTRACTORS = {
    'ABSOLUTE_CERTAINTY': Attractor(
        name='Pewność absolutna',
        coordinates=np.array([1.0, 1.0, 0.0]),
        basin_radius=0.15,
        stability_index=0.95,
        type='fixed_point'
    ),
    'CHAOS': Attractor(
        name='Chaos semantyczny',
        coordinates=np.array([0.0, 0.0, 1.0]),
        basin_radius=0.20,
        stability_index=0.10,
        type='strange_attractor'
    ),
    'BALANCED_DISCOURSE': Attractor(
        name='Dyskurs zrównoważony',
        coordinates=np.array([0.65, 0.70, 0.35]),
        basin_radius=0.25,
        stability_index=0.75,
        type='fixed_point'
    ),
    'LEGAL_NORM': Attractor(
        name='Norma prawna',
        coordinates=np.array([0.85, 0.80, 0.15]),
        basin_radius=0.18,
        stability_index=0.88,
        type='fixed_point'
    ),
    'POETIC_AMBIGUITY': Attractor(
        name='Wieloznaczność poetycka',
        coordinates=np.array([0.40, 0.55, 0.75]),
        basin_radius=0.30,
        stability_index=0.45,
        type='limit_cycle'
    ),
    'IRONIC_INVERSION': Attractor(
        name='Odwrócenie ironiczne',
        coordinates=np.array([0.25, 0.30, 0.85]),
        basin_radius=0.22,
        stability_index=0.35,
        type='strange_attractor'
    ),
    'IMPERATIVE_COMMAND': Attractor(
        name='Rozkaz imperatywny',
        coordinates=np.array([0.95, 0.65, 0.10]),
        basin_radius=0.12,
        stability_index=0.92,
        type='fixed_point'
    ),
}


class TopologicalAttractorAnalyzer:
    """Analyzes topological attractors in semantic D-S-E space."""

    def __init__(self, attractors: Dict[str, Attractor] = None):
        """
        Initialize attractor analyzer.

        Args:
            attractors: Dictionary of known attractors (uses default if None)
        """
        self.attractors = attractors or SEMANTIC_ATTRACTORS

    def find_nearest_attractor(self, coords: np.ndarray) -> Tuple[str, float, Dict]:
        """
        Find nearest attractor to given coordinates.

        Args:
            coords: Current coordinates [D, S, E]

        Returns:
            Attractor name, distance, and metadata
        """
        min_distance = float('inf')
        nearest_attractor = None
        nearest_name = None

        for name, attractor in self.attractors.items():
            distance = np.linalg.norm(coords - attractor.coordinates)

            if distance < min_distance:
                min_distance = distance
                nearest_attractor = attractor
                nearest_name = name

        # Check if within basin of attraction
        in_basin = bool(min_distance <= nearest_attractor.basin_radius)

        # Calculate pull strength (inverse distance, weighted by stability)
        if min_distance > 0:
            pull_strength = (nearest_attractor.stability_index / min_distance) * \
                          (1.0 if in_basin else 0.5)
        else:
            pull_strength = 1.0

        metadata = {
            'attractor_name': nearest_attractor.name,
            'attractor_type': nearest_attractor.type,
            'distance': float(min_distance),
            'in_basin': in_basin,
            'basin_radius': float(nearest_attractor.basin_radius),
            'pull_strength': float(np.clip(pull_strength, 0, 1)),
            'stability_index': float(nearest_attractor.stability_index),
            'coordinates': nearest_attractor.coordinates.tolist()
        }

        return nearest_name, float(min_distance), metadata

    def calculate_trajectory(self,
                           start_coords: np.ndarray,
                           end_coords: np.ndarray,
                           steps: int = 10) -> Dict:
        """
        Calculate trajectory between two points in D-S-E space.

        Args:
            start_coords: Starting coordinates [D, S, E]
            end_coords: Ending coordinates [D, S, E]
            steps: Number of interpolation steps

        Returns:
            Trajectory data with attractors encountered
        """
        # Linear interpolation path
        trajectory_points = []
        attractors_encountered = []

        for i in range(steps + 1):
            t = i / steps
            point = start_coords + t * (end_coords - start_coords)
            trajectory_points.append(point)

            # Check for attractor proximity
            nearest_name, distance, metadata = self.find_nearest_attractor(point)

            if metadata['in_basin'] and (not attractors_encountered or
                                        attractors_encountered[-1]['name'] != nearest_name):
                attractors_encountered.append({
                    'step': i,
                    'name': nearest_name,
                    'distance': distance,
                    'position': point.tolist()
                })

        # Calculate path length
        path_length = 0.0
        for i in range(1, len(trajectory_points)):
            path_length += np.linalg.norm(trajectory_points[i] - trajectory_points[i-1])

        # Calculate curvature (deviation from straight line)
        direct_distance = np.linalg.norm(end_coords - start_coords)
        curvature = path_length - direct_distance if path_length > direct_distance else 0.0

        return {
            'start': start_coords.tolist(),
            'end': end_coords.tolist(),
            'path_length': float(path_length),
            'direct_distance': float(direct_distance),
            'curvature': float(curvature),
            'trajectory_points': [p.tolist() for p in trajectory_points],
            'attractors_encountered': attractors_encountered,
            'num_attractors': len(attractors_encountered)
        }

    def analyze_basin_of_attraction(self, coords: np.ndarray) -> Dict:
        """
        Analyze which basin of attraction the point belongs to.

        Args:
            coords: Coordinates [D, S, E]

        Returns:
            Basin analysis data
        """
        all_attractors_distances = []

        for name, attractor in self.attractors.items():
            distance = np.linalg.norm(coords - attractor.coordinates)
            in_basin = bool(distance <= attractor.basin_radius)

            all_attractors_distances.append({
                'name': name,
                'display_name': attractor.name,
                'distance': float(distance),
                'in_basin': in_basin,
                'type': attractor.type,
                'stability': float(attractor.stability_index)
            })

        # Sort by distance
        all_attractors_distances.sort(key=lambda x: x['distance'])

        # Find all basins containing this point
        basins_containing = [a for a in all_attractors_distances if a['in_basin']]

        return {
            'primary_basin': basins_containing[0] if basins_containing else None,
            'all_basins': basins_containing,
            'num_basins': int(len(basins_containing)),
            'nearest_attractors': all_attractors_distances[:3],  # Top 3
            'is_in_basin': bool(len(basins_containing) > 0),
            'basin_overlap': bool(len(basins_containing) > 1)
        }


# =============================================================================
# TEMPORAL EVOLUTION & TRAJECTORY ANALYSIS
# =============================================================================

class TemporalEvolutionAnalyzer:
    """Analyzes temporal evolution of D-S-E coordinates."""

    def __init__(self, history_size: int = 100):
        """
        Initialize temporal evolution analyzer.

        Args:
            history_size: Maximum number of historical states to keep
        """
        self.history = []
        self.history_size = history_size
        self.attractor_analyzer = TopologicalAttractorAnalyzer()

    def add_state(self, coords: np.ndarray, timestamp: float = None):
        """
        Add new state to evolution history.

        Args:
            coords: Coordinates [D, S, E]
            timestamp: Optional timestamp (auto-generated if None)
        """
        import time

        if timestamp is None:
            timestamp = time.time()

        self.history.append({
            'coords': coords,
            'timestamp': timestamp
        })

        # Maintain history size limit
        if len(self.history) > self.history_size:
            self.history.pop(0)

    def calculate_velocity(self, window: int = 2) -> Optional[np.ndarray]:
        """
        Calculate velocity in D-S-E space.

        Args:
            window: Number of recent points to use

        Returns:
            Velocity vector or None if insufficient data
        """
        if len(self.history) < window:
            return None

        recent_states = self.history[-window:]

        # Calculate time difference
        dt = recent_states[-1]['timestamp'] - recent_states[0]['timestamp']

        if dt == 0:
            return np.array([0.0, 0.0, 0.0])

        # Calculate coordinate difference
        dx = recent_states[-1]['coords'] - recent_states[0]['coords']

        # Velocity = dx/dt
        velocity = dx / dt

        return velocity

    def calculate_acceleration(self, window: int = 3) -> Optional[np.ndarray]:
        """
        Calculate acceleration in D-S-E space.

        Args:
            window: Number of recent points to use

        Returns:
            Acceleration vector or None if insufficient data
        """
        if len(self.history) < window:
            return None

        recent_states = self.history[-window:]

        # Calculate two velocity vectors
        dt1 = recent_states[1]['timestamp'] - recent_states[0]['timestamp']
        dt2 = recent_states[2]['timestamp'] - recent_states[1]['timestamp']

        if dt1 == 0 or dt2 == 0:
            return np.array([0.0, 0.0, 0.0])

        v1 = (recent_states[1]['coords'] - recent_states[0]['coords']) / dt1
        v2 = (recent_states[2]['coords'] - recent_states[1]['coords']) / dt2

        # Acceleration = dv/dt
        dv = v2 - v1
        dt_avg = (dt1 + dt2) / 2

        acceleration = dv / dt_avg

        return acceleration

    def detect_bifurcation_points(self, threshold: float = 0.5) -> List[Dict]:
        """
        Detect bifurcation points (sudden direction changes).

        Args:
            threshold: Angle threshold for bifurcation (in radians)

        Returns:
            List of bifurcation points
        """
        if len(self.history) < 3:
            return []

        bifurcations = []

        for i in range(1, len(self.history) - 1):
            # Calculate vectors before and after point
            v_before = self.history[i]['coords'] - self.history[i-1]['coords']
            v_after = self.history[i+1]['coords'] - self.history[i]['coords']

            # Normalize
            norm_before = np.linalg.norm(v_before)
            norm_after = np.linalg.norm(v_after)

            if norm_before > 1e-6 and norm_after > 1e-6:
                v_before_norm = v_before / norm_before
                v_after_norm = v_after / norm_after

                # Calculate angle between vectors
                cos_angle = np.dot(v_before_norm, v_after_norm)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)

                # If angle exceeds threshold, it's a bifurcation
                if angle > threshold:
                    bifurcations.append({
                        'index': i,
                        'timestamp': self.history[i]['timestamp'],
                        'coords': self.history[i]['coords'].tolist(),
                        'angle': float(angle),
                        'angle_degrees': float(np.degrees(angle))
                    })

        return bifurcations

    def predict_next_state(self,
                          prediction_horizon: float = 1.0,
                          method: str = 'linear') -> Optional[Dict]:
        """
        Predict next state based on current trajectory.

        Args:
            prediction_horizon: Time ahead to predict
            method: Prediction method ('linear', 'accelerated')

        Returns:
            Predicted state or None if insufficient data
        """
        if len(self.history) < 2:
            return None

        current_state = self.history[-1]
        current_coords = current_state['coords']
        current_time = current_state['timestamp']

        if method == 'linear':
            velocity = self.calculate_velocity(window=2)
            if velocity is None:
                return None

            predicted_coords = current_coords + velocity * prediction_horizon

        elif method == 'accelerated':
            velocity = self.calculate_velocity(window=2)
            acceleration = self.calculate_acceleration(window=3)

            if velocity is None or acceleration is None:
                return None

            # x(t) = x0 + v*t + 0.5*a*t²
            predicted_coords = current_coords + \
                             velocity * prediction_horizon + \
                             0.5 * acceleration * (prediction_horizon ** 2)
        else:
            raise ValueError(f"Unknown prediction method: {method}")

        # Clip to valid range [0, 1]
        predicted_coords = np.clip(predicted_coords, 0.0, 1.0)

        # Find nearest attractor for predicted position
        nearest_name, distance, metadata = \
            self.attractor_analyzer.find_nearest_attractor(predicted_coords)

        return {
            'predicted_coords': predicted_coords.tolist(),
            'predicted_timestamp': current_time + prediction_horizon,
            'prediction_method': method,
            'nearest_attractor': metadata,
            'confidence': self._calculate_prediction_confidence()
        }

    def _calculate_prediction_confidence(self) -> float:
        """
        Calculate confidence in prediction based on trajectory stability.

        Returns:
            Confidence score [0, 1]
        """
        if len(self.history) < 3:
            return 0.5

        # Calculate variance in recent velocities
        recent_velocities = []
        for i in range(len(self.history) - 2, max(0, len(self.history) - 6), -1):
            dt = self.history[i+1]['timestamp'] - self.history[i]['timestamp']
            if dt > 0:
                v = (self.history[i+1]['coords'] - self.history[i]['coords']) / dt
                recent_velocities.append(v)

        if not recent_velocities:
            return 0.5

        # Low variance = high confidence
        velocity_variance = np.var(recent_velocities, axis=0).mean()
        confidence = np.exp(-velocity_variance)  # Exponential decay

        return float(np.clip(confidence, 0.0, 1.0))

    def get_evolution_summary(self) -> Dict:
        """
        Get summary of temporal evolution.

        Returns:
            Evolution summary with key metrics
        """
        if not self.history:
            return {'error': 'No history available'}

        coords_array = np.array([s['coords'] for s in self.history])

        # Calculate statistics
        mean_coords = coords_array.mean(axis=0)
        std_coords = coords_array.std(axis=0)

        # Calculate total distance traveled
        total_distance = 0.0
        for i in range(1, len(self.history)):
            total_distance += np.linalg.norm(
                self.history[i]['coords'] - self.history[i-1]['coords']
            )

        # Time span
        time_span = self.history[-1]['timestamp'] - self.history[0]['timestamp']

        # Current state
        current_coords = self.history[-1]['coords']
        nearest_name, distance, attractor_metadata = \
            self.attractor_analyzer.find_nearest_attractor(current_coords)

        # Velocity and acceleration
        velocity = self.calculate_velocity()
        acceleration = self.calculate_acceleration()

        # Bifurcations
        bifurcations = self.detect_bifurcation_points()

        return {
            'num_states': len(self.history),
            'time_span': float(time_span),
            'total_distance': float(total_distance),
            'mean_coords': mean_coords.tolist(),
            'std_coords': std_coords.tolist(),
            'current_coords': current_coords.tolist(),
            'current_attractor': attractor_metadata,
            'velocity': velocity.tolist() if velocity is not None else None,
            'acceleration': acceleration.tolist() if acceleration is not None else None,
            'num_bifurcations': len(bifurcations),
            'bifurcation_points': bifurcations
        }


# =============================================================================
# INTEGRATION FUNCTION
# =============================================================================

def analyze_topological_context(coords: np.ndarray,
                               previous_coords: Optional[np.ndarray] = None,
                               temporal_window: Optional[List[np.ndarray]] = None) -> Dict:
    """
    Complete topological and temporal analysis.

    Args:
        coords: Current coordinates [D, S, E]
        previous_coords: Previous coordinates (for trajectory)
        temporal_window: List of recent coordinates for evolution analysis

    Returns:
        Complete topological analysis
    """
    attractor_analyzer = TopologicalAttractorAnalyzer()

    # Basic attractor analysis
    nearest_name, distance, attractor_metadata = \
        attractor_analyzer.find_nearest_attractor(coords)

    # Basin analysis
    basin_analysis = attractor_analyzer.analyze_basin_of_attraction(coords)

    result = {
        'current_position': {
            'coords': coords.tolist(),
            'nearest_attractor': attractor_metadata,
            'basin_analysis': basin_analysis
        }
    }

    # Trajectory analysis if previous state available
    if previous_coords is not None:
        trajectory = attractor_analyzer.calculate_trajectory(
            previous_coords, coords, steps=10
        )
        result['trajectory'] = trajectory

    # Temporal evolution if window available
    if temporal_window is not None and len(temporal_window) > 0:
        evolution_analyzer = TemporalEvolutionAnalyzer()

        # Add historical states
        for i, coord_state in enumerate(temporal_window):
            evolution_analyzer.add_state(coord_state, timestamp=float(i))

        # Add current state
        evolution_analyzer.add_state(coords, timestamp=float(len(temporal_window)))

        # Get evolution summary
        evolution_summary = evolution_analyzer.get_evolution_summary()
        result['temporal_evolution'] = evolution_summary

        # Prediction
        prediction = evolution_analyzer.predict_next_state(
            prediction_horizon=1.0,
            method='accelerated'
        )
        if prediction:
            result['prediction'] = prediction

    return result


if __name__ == "__main__":
    print("GTMØ Topological Attractors & Temporal Evolution Module")
    print("=" * 60)

    # Test attractor detection
    analyzer = TopologicalAttractorAnalyzer()

    test_coords = [
        ([0.85, 0.80, 0.15], "Legal norm"),
        ([0.25, 0.30, 0.85], "Ironic inversion"),
        ([0.65, 0.70, 0.35], "Balanced discourse"),
    ]

    for coords, expected in test_coords:
        coords_arr = np.array(coords)
        name, dist, metadata = analyzer.find_nearest_attractor(coords_arr)
        print(f"\nTest: {expected}")
        print(f"  Coords: {coords}")
        print(f"  Nearest: {metadata['attractor_name']}")
        print(f"  Distance: {dist:.3f}")
        print(f"  In basin: {metadata['in_basin']}")

    # Test temporal evolution
    print("\n" + "=" * 60)
    print("Temporal Evolution Test")
    print("=" * 60)

    evolution = TemporalEvolutionAnalyzer()

    # Simulate trajectory
    trajectory_points = [
        np.array([0.5, 0.5, 0.5]),
        np.array([0.6, 0.6, 0.4]),
        np.array([0.7, 0.7, 0.3]),
        np.array([0.75, 0.75, 0.25]),
        np.array([0.85, 0.80, 0.15]),  # Moving toward legal norm
    ]

    for i, point in enumerate(trajectory_points):
        evolution.add_state(point, timestamp=float(i))

    summary = evolution.get_evolution_summary()
    print(f"\nEvolution summary:")
    print(f"  Total distance: {summary['total_distance']:.3f}")
    print(f"  Mean coords: {[f'{x:.3f}' for x in summary['mean_coords']]}")
    print(f"  Current attractor: {summary['current_attractor']['attractor_name']}")

    if summary['velocity']:
        print(f"  Velocity: {[f'{x:.3f}' for x in summary['velocity']]}")

    prediction = evolution.predict_next_state()
    if prediction:
        print(f"\nPredicted next state:")
        print(f"  Coords: {[f'{x:.3f}' for x in prediction['predicted_coords']]}")
        print(f"  Confidence: {prediction['confidence']:.3f}")
        print(f"  Nearest attractor: {prediction['nearest_attractor']['attractor_name']}")

    print("\n" + "=" * 60)
