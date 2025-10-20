import sys
import pandas as pd
import uuid
from botorch.models import SingleTaskGP
from botorch.acquisition import qLogExpectedImprovement, qExpectedImprovement 
from ax import Client
from src import ax_helper
import numpy as np
from src.gp_and_acq_f import BocoliClassLoader

class BayesClientManager:

    def __init__(
        self,
        data: pd.DataFrame,
        feature_labels: list[str],
        
        bounds: dict,
        response_label: str = 'response',
    ) -> None:
        self.feature_labels = feature_labels
        self.response_label = response_label

        loader = BocoliClassLoader()
        # Use the richer info dicts (name -> {class,type,description})
        self.gp_options = loader.gaussian_process_info
        self.acq_f_options = loader.acquisition_function_info

        self.group_label = "group"
        self.id_label = "unique_id"
        self.data = self._preprocess_data(data)

        self.bounds = self._preprocess_bounds(bounds)

        self.look_coords = np.zeros(len(feature_labels))

        self._objective_direction = "maximise"  # or "minimise"

        self.resync_self = None

    @staticmethod
    @property
    def gp_options():
        """overwrites to dict once instantiated to allow access without instantiation"""
        loader = BocoliClassLoader()
        return loader.gaussian_process_info
    
    @staticmethod
    @property
    def acq_f_options():
        """overwrites to dict once instantiated to allow access without instantiation"""
        loader = BocoliClassLoader()
        return loader.acquisition_function_info


    @property
    def objective_direction(self):
        return self._objective_direction
    
    @objective_direction.setter
    def objective_direction(self, direction: str):
        if direction.lower() not in ["maximise", "minimise"]:
            raise ValueError("Objective direction must be 'maximise' or 'minimise'")
        self._objective_direction = direction.lower()

    @property
    def _ax_objective_direction(self):
        if self.objective_direction == "maximise":
            return "-loss"
        elif self.objective_direction == "minimise":
            return "loss"

    @staticmethod
    def _generate_id():
        return str(uuid.uuid4())[:8]

    def _preprocess_data(self, data: pd.DataFrame):
        """Preprocess data for loading"""
        if self.response_label not in data.columns:
            data[self.response_label] = np.nan
        elif not all(label in data.columns for label in self.feature_labels):
            missing = [
                label for label in self.feature_labels if label not in data.columns
            ]
            raise ValueError(f"Feature labels {missing} not found in data columns.")

        def generate_group_labels(df: pd.DataFrame):
            """Generate group labels based on feature combinations"""
            df[self.group_label] = df.groupby(self.feature_labels, sort=False).ngroup()
            return df

        def generate_id_labels(df: pd.DataFrame):

            # Generate shorter unique IDs (first 8 characters of UUID)
            ids = [self._generate_id() for _ in range(len(df))]
            df[self.id_label] = ids
            return df

        if self.group_label not in data.columns:
            data = generate_group_labels(data)

        if self.id_label not in data.columns:
            generate_id_labels(data)

        # Convert group column to int, handling NaN values
        data[self.group_label] = pd.to_numeric(data[self.group_label], errors='coerce').fillna(0).astype(int)

        return data

    def _preprocess_bounds(self, bounds: dict):
        """Preprocess bounds for Bayesian optimization"""
        if bounds is None:
            return None
            
        if any(label not in self.feature_labels for label in bounds.keys()):
            missing = [
                label for label in bounds.keys() if label not in self.feature_labels
            ]
            raise ValueError(f"Bounds specified for unknown features: {missing}")

        bounds = self._check_bound_structure(bounds)
        return bounds

    def _check_bound_structure(self, bounds: dict) -> dict[dict]:
        """Check the structure of bounds for Bayesian optimization
        Bounds must be structured as follows:
        {
            "feature1": {"lower_bound": float, "upper_bound": float, "log_scale": bool},
            "feature2": {"lower_bound": float, "upper_bound": float, "log_scale": bool},
            ...
        }
        """
        if any(label not in self.feature_labels for label in bounds.keys()):
            missing = [
                label for label in bounds.keys() if label not in self.feature_labels
            ]
            raise ValueError(f"Bounds specified for unknown features: {missing}")

        for label in bounds.keys():
            bound_config = bounds[label]
            if not isinstance(bound_config, dict):
                raise ValueError(
                    f"Bounds for feature '{label}' must be a dictionary with 'lower', 'upper', and 'log' keys."
                )

            required_keys = {"lower_bound", "upper_bound", "log_scale"}
            if not all(key in bound_config for key in required_keys):
                missing_keys = required_keys - set(bound_config.keys())
                raise ValueError(
                    f"Bounds for feature '{label}' missing required keys: {missing_keys}"
                )

            low, high, log = (
                bound_config["lower_bound"],
                bound_config["upper_bound"],
                bound_config["log_scale"],
            )

            if not (isinstance(low, (int, float)) and isinstance(high, (int, float))):
                raise ValueError(
                    f"Bounds for feature '{label}' must have numeric 'lower' and 'upper' values."
                )
            if low >= high:
                raise ValueError(
                    f"Invalid bounds for feature '{label}': lower {low} must be less than upper {high}."
                )
            if not isinstance(log, bool):
                raise ValueError(
                    f"Log scaling for feature '{label}' must be a boolean value."
                )

        return bounds

    @property
    def has_response(self):
        if self.resync_self is not None:
            self.resync_self()
        if self.data is None or self.data.empty:
            return False
        return not self.data[self.response_label].isna().all()

    @property
    def gp(self):
        """Initialize and return the Gaussian Process model"""
        if not hasattr(self, "_gp"):
            # take the first configured GP info and use its class

            first = next(iter(self.gp_options.values()))
            self._gp = first["class"]
        return self._gp

    @gp.setter
    def gp(self, gp_name: str):
        if gp_name not in self.gp_options:
            raise ValueError(
                f"GP model '{gp_name}' not recognized. Available models: {list(self.gp_options.keys())}"
            )
        self._gp = self.gp_options[gp_name]["class"]

    @property
    def acquisition_function(self):
        """Initialize and return the acquisition function"""
        if not hasattr(self, "_acquisition_function"):
            first = next(iter(self.acq_f_options.values()))
            self._acquisition_function = first["class"]
        return self._acquisition_function

    @acquisition_function.setter
    def acquisition_function(self, acq_f_name: str):
        if acq_f_name not in self.acq_f_options:
            raise ValueError(
                f"Acquisition function '{acq_f_name}' not recognized. Available functions: {list(self.acq_f_options.keys())}"
            )
        self._acquisition_function = self.acq_f_options[acq_f_name]["class"]

    @property
    def X(self):
        return self.data[self.feature_labels].to_numpy()

    @property
    def Y(self):
        return self.data[[self.response_label]].to_numpy()

    @property
    def observations(self):
        return self.data[
            self.feature_labels + [self.response_label, self.group_label, self.id_label]
        ].dropna(columns=self.response_label)

    @property
    def _ax_parameters(self):
        from ax import RangeParameterConfig

        if self.bounds is None:
            raise ValueError("Bounds must be specified to create Ax parameters")

        return [
            RangeParameterConfig(
                name=label,
                parameter_type="float",
                bounds=(bound_config["lower_bound"], bound_config["upper_bound"]),
                scaling="log_scale" if bound_config["log_scale"] else "linear",
            )
            for label, bound_config in self.bounds.items()
        ]

    @property
    def agg_stats(self):
        return (
            self.data.groupby([self.group_label] + self.feature_labels)[
                self.response_label
            ]
            .agg(["mean", "std", "count"])
            .reset_index()
        )



    def get_best_group(self):
        """Get the group label of the best-performing observation"""
        if self.agg_stats.empty:
            return None
        if self.objective_direction == "minimise":
            best_idx = self.agg_stats["mean"].idxmin()
        else:
            best_idx = self.agg_stats["mean"].idxmax()
        return self.agg_stats.loc[best_idx, self.group_label]

    def get_best_coordinates(self):
        """Compatibility helper: return coordinates dict for best-performing group.

        Returns None when there are no aggregated stats.
        """
        best_group = self.get_best_group()
        if best_group is None:
            return None
        return self.get_group_coords(int(best_group))

    def get_group_coords(self, group_label:int):
        """Get the coordinates for a given group label"""
        group_data = self.data[self.data[self.group_label] == group_label]
        if group_data.empty:
            return None
        return group_data[self.feature_labels].iloc[0].to_dict()

    def _create_ax_client(self):
        client = Client()

        client.configure_experiment(parameters=self._ax_parameters)
        client.configure_optimization(objective=self._ax_objective_direction)

        generation_strategy = ax_helper.get_full_strategy(
            gp=self.gp, acqf_class=self.acquisition_function
        )
        client.set_generation_strategy(generation_strategy=generation_strategy)

        return client

    def load_data_to_client(self):
        """Load existing data into the Ax Client"""

        client = self._create_ax_client()

        for _, row in self.data.iterrows():
            params = {label: row[label] for label in self.feature_labels}
            trial_index = client.attach_trial(parameters=params)
            if not np.isnan(row[self.response_label]):
                client.complete_trial(
                    trial_index=trial_index,
                    raw_data={self.response_label: row[self.response_label]},
                )

        return client

    def retrieve_data_from_client(self, client: Client):
        """Retrieve data from Ax Client into DataFrame"""
        df = ax_helper.get_obs_from_client(client)
        df = self._preprocess_data(df)
        return df

    @staticmethod
    def init_from_client(client: Client):
        """Initialize BayesClientManager from an existing Ax Client"""
        if not isinstance(client, Client):
            raise ValueError("Provided client is not an instance of ax.Client")

        df = ax_helper.get_obs_from_client(client)
        feature_labels = list(client._experiment.parameters.keys())
        response_label = list(client._experiment.metrics.keys())[0]

        """Extract bounds from client parameters"""
        client_bounds = list(client._experiment.parameters.values())
        bounds = {
            label: {"lower_bound": param.lower, "upper_bound": param.upper, "log_scale": param.log_scale}
            for label, param in zip(feature_labels, client_bounds)
        }

        manager = BayesClientManager(
            data=df,
            feature_labels=feature_labels,
            response_label=response_label,
            bounds=bounds,
        )

        return manager


    def get_batch_targets(self, n_targets: int):
        """Get next batch of target points from the client"""
        # Store original data count to identify new trials
        original_count = len(self.data)
        
        client = self.load_data_to_client()
        client.get_next_trials(max_trials=n_targets)

        # Get the updated data from client (this includes new targets)
        raw_new_data = self.retrieve_data_from_client(client)

        combined_data = []
        
        # Create a list to track which original rows have been matched
        original_data_used = set()
        
        # Get the next available group number for new trials
        max_existing_group = self.data[self.group_label].max() if len(self.data) > 0 else -1
        next_group_number = max_existing_group + 1
        
        for idx, new_row in raw_new_data.iterrows():
            # Try to find matching row in original data by parameters
            matching_mask = True
            for param in self.feature_labels:
                matching_mask &= (self.data[param] == new_row[param])
            
            existing_matches = self.data[matching_mask]
            
            # Find an unused matching row (to avoid duplicating responses)
            matched_row = None
            for _, existing_row in existing_matches.iterrows():
                row_index = existing_row.name
                if row_index not in original_data_used:
                    matched_row = existing_row.copy()
                    original_data_used.add(row_index)
                    # Keep the original group label - don't overwrite it
                    break
            
            if matched_row is not None:
                # Use existing row data (preserves unique_id, response, and group)
                combined_data.append(matched_row)
            else:
                # This is a new trial - assign next available group number
                new_row[self.group_label] = next_group_number
                next_group_number += 1
                combined_data.append(new_row)
        
        # Convert back to DataFrame
        self.data = pd.DataFrame(combined_data).reset_index(drop=True)
        return self.data

    def complete_trial_by_id(self, unique_id, response_value):
        index = self.data[self.data[self.id_label] == unique_id].index
        self.data.loc[index, self.response_label] = response_value

    @property
    def pending_targets(self):
        return self.data[self.data[self.response_label].isna()]
    

    def change_response_by_id(self, unique_id, new_response):
        index = self.data[self.data[self.id_label] == unique_id].index
        self.data.loc[index, self.response_label] = new_response

    def delete_by_id(self, unique_id: str) -> None:
        """Remove entry by unique ID and reset DataFrame index."""
        matching_indices = self.data[self.data[self.id_label] == unique_id].index
        if not matching_indices.empty:
            self.data = self.data.drop(matching_indices).reset_index(drop=True)

    def _get_group_for_coords(self, coords: list[float]) -> int:
        """Get the group label for a given set of coordinates."""
        if len(coords) != len(self.feature_labels):
            raise ValueError(
                f"Coordinates length {len(coords)} does not match number of features {len(self.feature_labels)}"
            )
        

        feature_data = self.data[self.feature_labels].to_numpy()
        matching_mask = np.all(feature_data == coords, axis=1)
        matching_rows = self.data[matching_mask]
        
        return (matching_rows.iloc[0][self.group_label] 
                if not matching_rows.empty 
                else self.data[self.group_label].max() + 1)

    def add_new_entry(self, entry: pd.DataFrame) -> None:
        """Add a new entry to the dataset with proper validation and preprocessing."""
        # Handle transposed data format
        if self.feature_labels[0] not in entry.columns:
            entry = entry.T
        
        # Validate required columns
        required_cols = self.feature_labels + [self.response_label]
        missing_cols = [col for col in required_cols if col not in entry.columns]
        if missing_cols:
            raise ValueError(f"Entry is missing required columns: {missing_cols}")
        
        # Auto-generate group if not provided
        if self.group_label not in entry.columns:
            coords = entry[self.feature_labels].iloc[0].tolist()
            entry[self.group_label] = self._get_group_for_coords(coords)
        
        # Auto-generate ID if not provided
        if self.id_label not in entry.columns:
            entry[self.id_label] = self._generate_id()
        
        # Add entry to dataset
        self.data = pd.concat([self.data, entry], ignore_index=True)

    @property
    def current_group_labels(self) -> list[int]:
        return sorted(self.data[self.group_label].dropna().unique().tolist())

    def get_groups(self) -> dict[int, pd.DataFrame]:
        """Get a dictionary of unique group labels and their corresponding data."""
        return {label: self.data[self.data[self.group_label] == label] for label in self.current_group_labels}
    
    @staticmethod
    def init_self_from_pickle(file:bytes):
        """Initialize BayesClientManager from a pickle file, handling module path changes."""
        return CompatibleUnpickler(file).load()
    
    @property
    def experiment_name(self):
        return getattr(self, "_experiment_name", "new_experiment")
    

import pickle


class CompatibleUnpickler(pickle.Unpickler):
    """Custom unpickler to handle module path changes"""
    def find_class(self, module, name):
        # Handle BayesClientManager from different module paths
        if name == 'BayesClientManager':
            return BayesClientManager
        
        # Handle module path remapping
        module_remapping = {
            '__main__': 'src.BayesClientManager',
            'BayesClientManager': 'src.BayesClientManager',
            'src.ui.UI_main': 'src.BayesClientManager'
        }
        
        if module in module_remapping and name == 'BayesClientManager':
            try:
                target_module = sys.modules[module_remapping[module]]
                return getattr(target_module, name)
            except (KeyError, AttributeError):
                pass
        
        return super().find_class(module, name)



        

        
        
  



def example_manager():
    df = pd.DataFrame(
        {
            "x1": [0.1, 0.4, 0.5, 0.7, 0.1],
            "x2": [1.0, 0.9, 0.8, 0.6, 1.0],
            "y": [0.5, 0.6, 0.55, np.nan, 0.45],
        }
    )
    feature_labels = ["x1", "x2"]
    response_label = "y"
    bounds = {
        "x1": {"lower_bound": 0.0, "upper_bound": 1.0, "log_scale": False},
        "x2": {"lower_bound": 0.5, "upper_bound": 1.5, "log_scale": True},
    }

    manager = BayesClientManager(
        data=df,
        feature_labels=feature_labels,
        response_label=response_label,
        bounds=bounds,
    )

    manager.gp = "HeteroWhiteSGP"
    manager.acquisition_function = "qLogExpectedImprovement"
    
    return manager




if __name__ == "__main__":
    manger = example_manager()
    import pickle
    pickle.dump(manger, open("data/example_manager.pkl", "wb"))
    print("Example manager saved to data/example_manager.pkl")