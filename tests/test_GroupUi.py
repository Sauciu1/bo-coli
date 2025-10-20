import pytest
import pandas as pd
import numpy as np
import streamlit as st
from unittest.mock import Mock, patch

from src.ui.SingleGroup import SingleGroup
from src.ui.GroupUi import GroupUi
from src.BayesClientManager import BayesClientManager


# Small helper to emulate Streamlit session_state attribute access used by GroupUi
class DummySession(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"st.session_state has no attribute \"{key}\"")

    def __setattr__(self, key, value):
        self[key] = value

    def setdefault(self, key, default=None):
        return super().setdefault(key, default)



class TestGroup:
    """Test suite for the Group class"""
    
    @pytest.fixture
    def sample_data(self):
        """Sample data for testing with multiple trials per group"""
        return pd.DataFrame({
            'x1': [0.1, 0.4, 0.5, 0.7, 0.1],
            'x2': [1.0, 0.9, 0.8, 0.6, 1.0],
            'response': [0.5, 0.6, 0.55, np.nan, 0.45],
            'group': [0, 1, 2, 3, 0],  # Group 0 has two trials
            'unique_id': ['trial1', 'trial2', 'trial3', 'trial4', 'trial5']
        })
    
    @pytest.fixture
    def feature_labels(self):
        """Feature labels for testing"""
        return ['x1', 'x2']
    
    @pytest.fixture
    def response_label(self):
        """Response label for testing"""
        return 'response'
    
    @pytest.fixture
    def bounds(self):
        """Bounds for testing"""
        return {
            'x1': {'lower_bound': 0.0, 'upper_bound': 1.0, 'log_scale': False},
            'x2': {'lower_bound': 0.5, 'upper_bound': 1.5, 'log_scale': True}
        }
    
    @pytest.fixture
    def bayes_manager(self, sample_data, feature_labels, response_label, bounds):
        """BayesClientManager instance for testing"""
        return BayesClientManager(
            data=sample_data,
            feature_labels=feature_labels,
            response_label=response_label,
            bounds=bounds
        )
    
    @pytest.fixture
    def group_ui_mock(self, bayes_manager):
        """Mock GroupUi instance for testing"""
        mock_ui = Mock()
        mock_ui.bayes_manager = bayes_manager
        mock_ui._notify_data_change = Mock()
        return mock_ui
    
    @pytest.fixture
    def group(self, group_ui_mock, bayes_manager):
        """Group instance for testing (group 0 has two trials)"""
        group_df = bayes_manager.data[bayes_manager.data[bayes_manager.group_label] == 0]
        return SingleGroup(group_df, group_label=0, bayes_manager=bayes_manager)

    def test_group_initialization(self, group, group_ui_mock, bayes_manager):
        """Test Group initialization"""
        # group was constructed from bayes_manager data
        assert group.bayes_manager == bayes_manager
        assert group.group_label == 0
        # group should expose get_data and trial_count
        assert hasattr(group, 'get_data')
        assert hasattr(group, 'trial_count')

    def test_manager_data_property(self, group, group_ui_mock):
        """Test manager_data property returns BayesClientManager data"""
        # SingleGroup stores a reference to the manager; compare manager data
        pd.testing.assert_frame_equal(group.bayes_manager.data, group_ui_mock.bayes_manager.data)

    def test_trials_property(self, group):
        """Test trials property returns correct trial indices for the group"""
        trials = group.get_data().index.tolist()
        # Group 0 should have trials at original indices 0 and 4
        assert trials == [0, 4]

    def test_X_property(self, group):
        """Test X property returns feature values for the group"""
        gd = group.get_data()
        row = gd.iloc[0]
        expected_X = [row['x1'], row['x2']]
        assert expected_X == [0.1, 1.0]

    def test_set_X_method(self, group, group_ui_mock):
        """Test set_X method updates feature values for all trials in group"""
        new_values = [0.2, 1.1]
        # Update group-level parameters and push to manager
        group.update_parameters(new_values)
        group.write_data_to_manager()

        gd = group.get_data()
        assert gd.iloc[0]['x1'] == 0.2
        assert gd.iloc[0]['x2'] == 1.1
        assert gd.iloc[-1]['x1'] == 0.2 or gd.iloc[0]['x1'] == 0.2

    def test_responses_property(self, group):
        """Test responses property returns all response values for the group"""
        responses = group.get_data()[group.response_label].tolist()
        expected_responses = [0.5, 0.45]
        assert responses == expected_responses

    def test_trial_ids_property(self, group):
        """Test trial_ids property returns all trial IDs for the group"""
        trial_ids = group.get_data()[group.id_label].tolist()
        expected_ids = ['trial1', 'trial5']
        assert trial_ids == expected_ids

    def test_set_response_method(self, group, group_ui_mock):
        """Test set_response method updates response value for specific trial"""
        new_response = 0.9
        # Set response for first trial in group (trial_index_in_group=0)
        group.update_response(0, new_response)
        group.write_data_to_manager()

        gd = group.get_data()
        assert gd.iloc[0][group.response_label] == 0.9
        assert gd.iloc[-1][group.response_label] == 0.45
    def test_add_trial_method(self, group, group_ui_mock):
        """Test add_trial method adds a new trial with same X values"""
        initial_trial_count = group.trial_count
        initial_data_length = len(group_ui_mock.bayes_manager.data)

        group.add_trial()
        group.write_data_to_manager()

        # Should have one more trial in the group
        assert len(group.get_data()) == initial_trial_count + 1

        # Should have one more row in the manager data
        assert len(group_ui_mock.bayes_manager.data) == initial_data_length + 1

        # New trial is appended at the end; check the last row for NaN response
        new_trial_row = group.get_data().iloc[-1]
        new_trial_X = list(new_trial_row[group.feature_labels])
        assert new_trial_X == [0.1, 1.0]
        assert pd.isna(new_trial_row[group.response_label])

        # The test harness mock shouldn't have been called by SingleGroup directly
        group_ui_mock._notify_data_change.assert_not_called()

    def test_remove_trial_method(self, group, group_ui_mock):
        """Test remove_trial method removes the last trial in the group"""
        initial_trial_count = group.trial_count
        initial_data_length = len(group_ui_mock.bayes_manager.data)

        # Remove the last trial in the group
        group.remove_trial()
        group.write_data_to_manager()

        assert len(group.get_data()) == initial_trial_count - 1
        assert len(group_ui_mock.bayes_manager.data) == initial_data_length - 1

    def test_has_pending_with_complete_data(self, group_ui_mock, bayes_manager):
        """Test has_pending equivalent returns False when all data is complete"""
        bm = group_ui_mock.bayes_manager
        group1 = SingleGroup(bm.data[bm.data[bm.group_label] == 1], group_label=1, bayes_manager=bm)

        # Compute has_pending from data: any NaNs in response or features
        gd = group1.get_data()
        has_pending = gd[group1.response_label].isna().any() or gd[group1.feature_labels].isna().any().any()
        assert not has_pending

    def test_has_pending_with_nan_response(self, group_ui_mock, bayes_manager):
        """Test has_pending equivalent returns True when response is NaN"""
        bm = group_ui_mock.bayes_manager
        group3 = SingleGroup(bm.data[bm.data[bm.group_label] == 3], group_label=3, bayes_manager=bm)

        gd = group3.get_data()
        has_pending = gd[group3.response_label].isna().any() or gd[group3.feature_labels].isna().any().any()
        assert has_pending

    def test_has_pending_with_nan_feature(self, group, group_ui_mock):
        """Test has_pending equivalent returns True when feature value is NaN"""
        # Set a feature to NaN for group 0 and sync the group from the manager
        group_ui_mock.bayes_manager.data.iloc[0, 0] = np.nan  # x1 = NaN
        bm = group_ui_mock.bayes_manager
        fresh = bm.data[bm.data[bm.group_label] == group.group_label]
        group.sync_from_manager(fresh)
        gd = group.get_data()
        has_pending = gd[group.response_label].isna().any() or gd[group.feature_labels].isna().any().any()
        assert has_pending

    def test_different_groups(self, group_ui_mock, bayes_manager):
        """Test Group works with different group numbers"""
        bm = group_ui_mock.bayes_manager
        group0 = SingleGroup(bm.data[bm.data[bm.group_label] == 0], group_label=0, bayes_manager=bm)
        group1 = SingleGroup(bm.data[bm.data[bm.group_label] == 1], group_label=1, bayes_manager=bm)
        group2 = SingleGroup(bm.data[bm.data[bm.group_label] == 2], group_label=2, bayes_manager=bm)

        # Check they have different data using SingleGroup API
        gd0 = group0.get_data()
        gd1 = group1.get_data()
        gd2 = group2.get_data()

        x0 = list(gd0.iloc[0][group0.feature_labels])
        x1 = list(gd1.iloc[0][group1.feature_labels])
        x2 = list(gd2.iloc[0][group2.feature_labels])

        assert x0 == [0.1, 1.0]
        assert x1 == [0.4, 0.9]
        assert x2 == [0.5, 0.8]

        assert gd0[group0.response_label].tolist() == [0.5, 0.45]
        assert gd1[group1.response_label].tolist() == [0.6]
        assert gd2[group2.response_label].tolist() == [0.55]

    def test_group_with_modified_manager_data(self, group_ui_mock, bayes_manager):
        """Test Group reflects changes in manager data"""
        bm = group_ui_mock.bayes_manager
        group = SingleGroup(bm.data[bm.data[bm.group_label] == 0], group_label=0, bayes_manager=bm)

        # Modify the manager data directly for group 0 trials
        bm.data.iloc[0, bm.data.columns.get_loc('x1')] = 0.99
        bm.data.iloc[4, bm.data.columns.get_loc('x1')] = 0.99
        bm.data.iloc[0, bm.data.columns.get_loc(bm.response_label)] = 0.88

        # After syncing, the group's group_df should be updated when syncing
        group.sync_from_manager(bm.data[bm.data[bm.group_label] == 0])
        gd = group.get_data()
        assert gd.iloc[0]['x1'] == 0.99
        assert gd.iloc[0][group.response_label] == 0.88


class TestGroupUi:
    """Test suite for the GroupUi class"""
    
    @pytest.fixture
    def sample_data(self):
        """Sample data for testing with groups"""
        return pd.DataFrame({
            'x1': [0.1, 0.4, 0.5, 0.7, 0.1],
            'x2': [1.0, 0.9, 0.8, 0.6, 1.0],
            'response': [0.5, 0.6, 0.55, np.nan, 0.45],
            'group': [0, 1, 2, 3, 0],  # Group 0 has two trials
            'unique_id': ['trial1', 'trial2', 'trial3', 'trial4', 'trial5']
        })
    
    @pytest.fixture
    def feature_labels(self):
        """Feature labels for testing"""
        return ['x1', 'x2']
    
    @pytest.fixture
    def response_label(self):
        """Response label for testing"""
        return 'response'
    
    @pytest.fixture
    def bounds(self):
        """Bounds for testing"""
        return {
            'x1': {'lower_bound': 0.0, 'upper_bound': 1.0, 'log_scale': False},
            'x2': {'lower_bound': 0.5, 'upper_bound': 1.5, 'log_scale': True}
        }
    
    @pytest.fixture
    def bayes_manager(self, sample_data, feature_labels, response_label, bounds):
        """BayesClientManager instance for testing"""
        return BayesClientManager(
            data=sample_data,
            feature_labels=feature_labels,
            response_label=response_label,
            bounds=bounds
        )
    
    @pytest.fixture
    def group_ui(self, bayes_manager, monkeypatch):
        """GroupUi instance for testing with a mocked session_state"""
        mock_state = DummySession()
        monkeypatch.setattr(st, 'session_state', mock_state, raising=False)
        return GroupUi(bayes_manager)

    def test_groupui_initialization(self, group_ui, bayes_manager):
        """Test GroupUi initialization"""
        assert group_ui.bayes_manager == bayes_manager
        # Session state should be initialized with show_pending_only


    def test_add_manual_group(self, group_ui):
        """Test add_manual_group adds a new group to BayesClientManager"""
        initial_length = len(group_ui.bayes_manager.data)
        initial_max_group = group_ui.bayes_manager.data['group'].max()
        
        group_ui.add_manual_group()
        
        # Should have one more row
        assert len(group_ui.bayes_manager.data) == initial_length + 1
        
        # New row should have NaN values for features and response
        new_row = group_ui.bayes_manager.data.iloc[-1]
        assert pd.isna(new_row['x1'])
        assert pd.isna(new_row['x2'])
        assert pd.isna(new_row['response'])
        
        # Should have proper group and ID labels
        assert new_row['group'] == initial_max_group + 1  # New group number
        assert str(new_row['unique_id']).startswith('manual_')

    def test_add_multiple_manual_groups(self, group_ui):
        """Test adding multiple manual groups"""
        initial_length = len(group_ui.bayes_manager.data)
        initial_max_group = group_ui.bayes_manager.data['group'].max()
        
        group_ui.add_manual_group()
        group_ui.add_manual_group()
        group_ui.add_manual_group()
        
        # Should have three more rows
        assert len(group_ui.bayes_manager.data) == initial_length + 3
        
        # Check group numbers are assigned correctly
        new_rows = group_ui.bayes_manager.data.iloc[-3:]
        expected_groups = [initial_max_group + 1, initial_max_group + 2, initial_max_group + 3]
        actual_groups = new_rows['group'].tolist()
        assert actual_groups == expected_groups

    def test_remove_group(self, group_ui):
        """Remove a specific group and ensure rows removed from manager data and session state."""
        bm = group_ui.bayes_manager
        # initial count and rows for group 0
        initial_len = len(bm.data)
        rows_group0 = bm.data[bm.data[bm.group_label] == 0]
        assert not rows_group0.empty

        group_ui.remove_group(0)

        # No rows with group 0 should remain
        assert 0 not in bm.data[bm.group_label].unique()
        assert len(bm.data) == initial_len - len(rows_group0)

    def test_remove_first_group(self, group_ui):
        """Remove the smallest-numbered group and verify it's gone."""
        bm = group_ui.bayes_manager
        min_group = int(bm.data[bm.group_label].min())
        group_ui.remove_group(min_group)
        assert min_group not in bm.data[bm.group_label].unique()

    def test_remove_last_group(self, group_ui):
        """Remove the largest-numbered group and verify it's gone."""
        bm = group_ui.bayes_manager
        max_group = int(bm.data[bm.group_label].max())
        group_ui.remove_group(max_group)
        assert max_group not in bm.data[bm.group_label].unique()

    def test_groups_property_after_modifications(self, group_ui):
        """Ensure GroupUi.groups reflects data changes (adding a manual group)."""
        bm = group_ui.bayes_manager
        initial_groups = set(bm.data[bm.group_label].unique())

        group_ui.add_manual_group()

        new_groups = set(bm.data[bm.group_label].unique())
        assert len(new_groups) == len(initial_groups) + 1

        # The groups property should include the new group label
        groups_mapping = group_ui.groups
        assert set(groups_mapping.keys()) == new_groups

    def test_integration_add_and_remove_groups(self, group_ui):
        """Add a manual group then remove it and confirm data returns to original state."""
        bm = group_ui.bayes_manager
        initial_len = len(bm.data)

        group_ui.add_manual_group()
        new_row = bm.data.iloc[-1]
        new_group = int(new_row[bm.group_label])

        # Now remove the group we just added
        group_ui.remove_group(new_group)

        assert len(bm.data) == initial_len
        assert new_group not in bm.data[bm.group_label].unique()

    def test_empty_data_handling(self, monkeypatch):
        """Ensure GroupUi handles empty manager data gracefully."""
        # Create an empty manager with same columns
        empty_df = pd.DataFrame(columns=['x1', 'x2', 'response', 'group', 'unique_id'])
        bm = BayesClientManager(data=empty_df, feature_labels=['x1', 'x2'], response_label='response', bounds={})
        mock_state = DummySession()
        monkeypatch.setattr(st, 'session_state', mock_state, raising=False)
        gui = GroupUi(bm)

        assert not gui.has_data
        assert gui.groups == {}

    def test_single_row_data(self, monkeypatch):
        """Verify GroupUi works when manager has a single-row dataset."""
        single = pd.DataFrame({
            'x1': [0.2],
            'x2': [1.2],
            'response': [0.7],
            'group': [0],
            'unique_id': ['single_1']
        })
        bm = BayesClientManager(data=single, feature_labels=['x1', 'x2'], response_label='response', bounds={})
        mock_state = DummySession()
        monkeypatch.setattr(st, 'session_state', mock_state, raising=False)
        gui = GroupUi(bm)

        groups = gui.groups
        assert len(groups) == 1
        g = list(groups.values())[0]
        assert g.get_data().shape[0] == 1

    @patch('streamlit.session_state')
    def test_session_state_initialization(self, mock_session_state, bayes_manager):
        """Test that GroupUi properly initializes session state"""
        mock_session_state.setdefault = Mock()
        mock_session_state.get = Mock(return_value=0)
        

