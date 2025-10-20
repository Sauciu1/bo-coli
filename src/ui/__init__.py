"""
UI Module for BioKernel Bayesian Optimization Interface

This module provides a comprehensive Streamlit-based user interface for 
Bayesian optimization experiments, including experiment initialization,
group management, plotting, and data visualization.

Main Components:
- ExperimentInitialiser: Main UI coordinator for experiment management
- InitExperiment: Handles new experiment setup and configuration  
- GroupUi: Manages experiment groups and trials
- SingleGroup: Individual group management and editing
- UiBayesPlotter: Plotting and visualization components

Example Usage:
    from src.ui import ExperimentInitialiser, GroupUi, UiBayesPlotter
    from src.BayesClientManager import BayesClientManager
    
    # Initialize the experiment interface
    initializer = ExperimentInitialiser()
    
    # Create a manager and UI components
    manager = BayesClientManager(...)
    group_ui = GroupUi(manager)
    plotter = UiBayesPlotter(manager)
"""

# Main UI classes
from .main_ui import ExperimentInitialiser
from .InitExperiment import InitExperiment
from .GroupUi import GroupUi
from .SingleGroup import SingleGroup
from .BayesPlotter import BayesPlotter


# Export main classes for public API
__all__ = [
    'ExperimentInitialiser',
    'InitExperiment', 
    'GroupUi',
    'SingleGroup',
    'BayesPlotter',
    'gp_and_acq_f_loader',
]

# Version info
__version__ = '1.0.0'
__author__ = 'Povilas Sauciuvienas'