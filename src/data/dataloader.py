"""
Data loading utilities for test case datasets.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any

class Methods2TestLoader:
    """Loader for Methods2Test dataset."""

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)

    def load_sample_data(self, max_samples: int = 1000) -> pd.DataFrame:
        """Load a sample of the dataset."""
        data = []

        # Generate synthetic data for demo
        generator = SyntheticTestCaseGenerator()
        return generator.generate_sample_data(max_samples)

class SyntheticTestCaseGenerator:
    """Generate synthetic test cases for demonstration."""

    def __init__(self):
        self.templates = [
            "test_{action}_{component}_{condition}",
            "should_{behavior}_when_{context}",
            "verify_{aspect}_for_{scenario}"
        ]

        self.actions = ['create', 'update', 'delete', 'retrieve', 'validate']
        self.components = ['user', 'order', 'payment', 'product', 'account']
        self.conditions = ['valid_input', 'invalid_input', 'empty_data']
        self.behaviors = ['return_success', 'throw_exception', 'update_database']
        self.contexts = ['user_logged_in', 'invalid_permissions', 'system_down']
        self.aspects = ['functionality', 'performance', 'security']
        self.scenarios = ['happy_path', 'error_handling', 'boundary_conditions']

    def generate_sample_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate synthetic test case data."""
        np.random.seed(42)
        data = []

        for i in range(n_samples):
            template = np.random.choice(self.templates)

            if 'action' in template:
                test_name = template.format(
                    action=np.random.choice(self.actions),
                    component=np.random.choice(self.components),
                    condition=np.random.choice(self.conditions)
                )
            elif 'behavior' in template:
                test_name = template.format(
                    behavior=np.random.choice(self.behaviors),
                    context=np.random.choice(self.contexts)
                )
            else:
                test_name = template.format(
                    aspect=np.random.choice(self.aspects),
                    scenario=np.random.choice(self.scenarios)
                )

            data.append({
                'test_case_id': test_name,
                'test_description': test_name.replace('_', ' '),
                'combined_text': test_name.replace('_', ' ')
            })

        return pd.DataFrame(data)

def load_demo_data(max_samples: int = 1000) -> pd.DataFrame:
    """Load demo data for development."""
    generator = SyntheticTestCaseGenerator()
    return generator.generate_sample_data(max_samples)
