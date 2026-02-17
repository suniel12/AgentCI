"""
Configuration loader.

Loads and validates agentci.yaml configuration files.
"""

from .models import TestSuite

import yaml
import os
from .models import TestSuite

def load_config(path: str = "agentci.yaml") -> TestSuite:
    """Load and validate agentci.yaml configuration file."""
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    suite = TestSuite(**config_dict)
    
    # Resolve relative paths relative to the config file
    base_dir = os.path.dirname(os.path.abspath(path))
    
    for test in suite.tests:
        if test.golden_trace and not os.path.isabs(test.golden_trace):
            test.golden_trace = os.path.join(base_dir, test.golden_trace)
            
    return suite
