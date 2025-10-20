import streamlit as st
import pandas as pd
import numpy as np
import uuid


class SingleGroup:
    def __init__(self, group_df, group_label, bayes_manager):
        self.group_df = group_df.copy()  # Work with a copy to avoid modifying original
        self.group_label = group_label
        self.feature_labels = bayes_manager.feature_labels
        self.response_label = bayes_manager.response_label
        self.id_label = bayes_manager.id_label
        self.bayes_manager = bayes_manager
        self._data_modified = False
    
    @property
    def trial_count(self):
        return len(self.group_df)
    
    @property 
    def is_empty(self):
        return self.group_df.empty
    
    def sync_from_manager(self, fresh_data):
        """Update group data from manager without losing local changes"""
        if not self._data_modified:
            self.group_df = fresh_data.copy()
    
    def clear_modified_flag(self):
        """Clear the modified flag after successful sync"""
        self._data_modified = False

    def add_trial(self):
        new_row = self._create_new_trial()
        self._append_trial(new_row)
    
    def _create_new_trial(self):
        if not self.is_empty:
            new_row = self.group_df.iloc[0].copy()
            new_row[self.response_label] = np.nan
        else:
            new_row = {param: 0.0 for param in self.feature_labels}
            new_row.update({self.response_label: np.nan, "group": self.group_label})
        
        new_row[self.id_label] = f"trial_{str(uuid.uuid4())[:8]}"
        return new_row
    
    def _append_trial(self, new_row):
        trial_df = pd.DataFrame([new_row])
        self.group_df = pd.concat([self.group_df, trial_df], ignore_index=True)
        self.bayes_manager.data = pd.concat([self.bayes_manager.data, trial_df], ignore_index=True)
        self._data_modified = True

    def remove_trial(self):
        if self.trial_count > 1:
            self.group_df = self.group_df.iloc[:-1].reset_index(drop=True)
            self._data_modified = True

    def update_parameters(self, new_values):
        for param, value in zip(self.feature_labels, new_values):
            self.group_df[param] = value
        self._data_modified = True

    def update_response(self, trial_index, value):
        if trial_index < len(self.group_df):
            self.group_df.iloc[
                trial_index, self.group_df.columns.get_loc(self.response_label)
            ] = value
            self._data_modified = True

    @st.fragment
    def render(self):
        """Render the group UI and return True if data was modified"""
        data_changed = False
        
        st.markdown(f"### Group {self.group_label}")

        cols = st.columns([1.5, 2, 0.15, 0.02])

        with cols[0]:
            st.write("**Parameters:**")
            if not self.is_empty:
                # Show X values (assuming all trials in group have same X values)
                x_values = [
                    self.group_df.iloc[0][param] for param in self.feature_labels
                ]
                display_x = [0.0 if pd.isna(val) else val for val in x_values]

                edited_x = st.data_editor(
                    pd.DataFrame([display_x], columns=self.feature_labels),
                    num_rows="fixed",
                    column_config={
                        col: st.column_config.NumberColumn(
                            help="Parameter value", step=1e-10, format="%.6e"
                        )
                        for col in self.feature_labels
                    },
                    key=f"x_values_group_{self.group_label}_{hash(str(self.group_df.index.tolist()))}",
                    hide_index=True,
                )

                # Update parameters if changed
                if not edited_x.empty:
                    new_x = edited_x.iloc[0].tolist()
                    if new_x != display_x:
                        self.update_parameters(new_x)
                        data_changed = True
            else:
                # Empty group - show default parameters
                default_x = [0.0] * len(self.feature_labels)
                st.data_editor(
                    pd.DataFrame([default_x], columns=self.feature_labels),
                    num_rows="fixed",
                    key=f"x_values_group_{self.group_label}",
                    hide_index=True,
                    disabled=True,
                )

        with cols[1]:
            st.write("**Response Values:**")
            if not self.is_empty:
                responses = self.group_df[self.response_label].tolist()
                response_data = {
                    f"Trial {i+1}": [resp if not pd.isna(resp) else None]
                    for i, resp in enumerate(responses)
                }

                edited_responses = st.data_editor(
                    pd.DataFrame(response_data),
                    column_config={
                        f"Trial {i+1}": st.column_config.NumberColumn(
                            help=f"Response value for trial {i+1}",
                            step=1e-10,
                            format="%.6e",
                            default=None,
                        )
                        for i in range(len(responses))
                    },
                    key=f"responses_group_{self.group_label}_{hash(str(self.group_df.index.tolist()))}",
                    hide_index=True,
                    num_rows="fixed",
                )

                # Update responses if changed
                if not edited_responses.empty:
                    for i, col_name in enumerate(edited_responses.columns):
                        new_val = edited_responses.iloc[0][col_name]
                        orig_val = responses[i]
                        if pd.isna(new_val) and not pd.isna(orig_val):
                            self.update_response(i, np.nan)
                            data_changed = True
                        elif not pd.isna(new_val) and (
                            pd.isna(orig_val) or new_val != orig_val
                        ):
                            self.update_response(i, new_val)
                            data_changed = True
            else:
                st.info("No trials in this group")

        with cols[2]:
            if st.button(
                "➕",
                key=f"add_trial_group_{self.group_label}_{hash(str(self.group_df.index.tolist()))}",
                help="Add trial",
                width='content'
            ):
                self.add_trial()
                data_changed = True
                st.rerun(scope="fragment")

            if self.trial_count > 1 and st.button(
                "➖",
                key=f"remove_trial_group_{self.group_label}_{hash(str(self.group_df.index.tolist()))}",
                help="Remove last trial",
                width='content'
            ):
                self.remove_trial()
                data_changed = True
                st.rerun(scope="fragment")
        
        return data_changed


    def get_data(self):
        return self.group_df.copy()

    def write_data_to_manager(self):
        """Efficiently update manager data only if modified"""
        if self._data_modified:
            # Use pandas update/merge approach instead of delete-and-append
            mask = self.bayes_manager.data[self.bayes_manager.group_label] != self.group_label
            other_data = self.bayes_manager.data[mask]
            self.bayes_manager.data = pd.concat([other_data, self.group_df], ignore_index=True)
            self.clear_modified_flag()


if __name__ == "__main__":




    st.set_page_config(layout="wide")
    st.title("Single Group Example")

    # Lazily import example_manager to avoid import-time side effects
    from src.BayesClientManager import example_manager

    # Initialize session state
    if "bayes_manager" not in st.session_state:
        st.session_state.bayes_manager = example_manager()
    if "groups" not in st.session_state:
        st.session_state.groups = {}

    # Render all groups
    for group_label, group_df in st.session_state.bayes_manager.get_groups().items():
        if group_label not in st.session_state.groups:
            st.session_state.groups[group_label] = SingleGroup(
                group_df, group_label, st.session_state.bayes_manager
            )

        if st.session_state.groups[group_label].render():
            st.session_state.groups[group_label].write_data_to_manager()
        st.divider()


    if st.button("Get All Group Data"):
        for group_label, group in st.session_state.groups.items():
            st.write(f"**Group {group_label}:**")
            st.dataframe(group.get_data(), width="stretch")
