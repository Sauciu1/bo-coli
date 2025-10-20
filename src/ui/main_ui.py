from os import write
import streamlit as st
import sys
import pickle
import json
import sys

from src.ui.InitExperiment import InitExperiment
from src.BayesClientManager import BayesClientManager
from src.ui.GroupUi import GroupUi
from src.ui.BayesPlotter import BayesPlotter
import time
import pathlib

class ExperimentInitialiser:
    def _init_or_load_exp(self):
       
        if st.session_state.get("initializing_experiment", False):
            return self._init_experiment()


        st.write("Initialize a new experiment or load an existing one.")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Initialize New Experiment", use_container_width=True):
                st.session_state.initializing_experiment = True
                st.rerun()

        with col2:
            self._load_exp_from_pickle_ui()
            # File upload directly visible

    def _load_exp_from_pickle_ui(self):
        uploaded_file = st.file_uploader("Upload saved experiment file", type=["pkl"])
        if uploaded_file is not None:
            try:

                uploaded_file.seek(0)

                manager = BayesClientManager.init_self_from_pickle(uploaded_file)

                if not isinstance(manager, BayesClientManager):
                    raise ValueError(
                        "The loaded object is not a BayesClientManager instance."
                    )

                # Store in session state
                st.session_state.bayes_manager = manager
                st.session_state.experiment_created = True
                st.success("✅ Experiment loaded successfully!")
                st.rerun()

            except (pickle.UnpicklingError, AttributeError, ModuleNotFoundError) as e:
                st.error(f"❌ Pickle file error: {e}")
            except Exception as e:
                st.error(f"❌ Failed to load file: {e}")

    def _init_experiment(self):
        if "new_exp" not in st.session_state:
            st.session_state.new_exp = InitExperiment()

        st.session_state.new_exp.create_experiment()

        if st.session_state.get(
            "experiment_configured", False
        ) and st.session_state.get("bayes_manager"):
            st.success("✅ Experiment configuration created!")
            if st.button(
                "Finish Setup and Start Experiment",
                type="primary",
                use_container_width=True,
            ):
                st.session_state.experiment_created = True
                st.session_state.initializing_experiment = False
                st.balloons()  # Show celebration immediately
                st.rerun()


class main_manager:
    def __init__(self):
        self.loader = ExperimentInitialiser()

    def main_loop(self):
        """Handles the main UI loop for experiment management"""
        st.title("BioKernel: Bayesian Optimization for Biological Systems")
        if not st.session_state.get("experiment_created", False):
            self.loader._init_or_load_exp()
        else:
            self.run_group_manager()
            BayesPlotter(self.bayes_manager).main_loop()

            st.divider()
            self._download_experiment()

        self.write_footnote()

    @property
    def group_manager(self):
        """Lazily initialize GroupUi"""
        if "group_manager" not in st.session_state and st.session_state.get(
            "bayes_manager"
        ):
            st.session_state.group_manager = GroupUi(st.session_state.bayes_manager)
        return st.session_state.get("group_manager")
    
    @property
    def bayes_manager(self):
        return st.session_state.get("bayes_manager", None)

    def run_group_manager(self):
        """Render the lements in group manager UI"""
        self.bayes_manager.sync_self = self.group_manager.sync_all_groups_to_manager


        self.group_manager.render_all()
        # self.group_manager.show_data_stats()

        st.divider()
        st.subheader("Visualization & Analysis")
        
        self.group_manager.sync_all_groups_to_manager()
  
    def _download_experiment(self):

        # Function to prepare download data with sync
        def prepare_download_data():
            # Sync all groups to manager when download is requested
            self.group_manager.sync_all_groups_to_manager()
            return pickle.dumps(self.group_manager.bayes_manager)

        current_time = time.strftime("%Y%m%d_%H%M")

        # Direct download button with on-demand data preparation
        st.download_button(
            label="💾 Download Experiment Data",
            data=prepare_download_data(),
            file_name=f"BioKernel_{self.bayes_manager.experiment_name}_{current_time}.pkl",
            mime="application/octet-stream",
            help="Download the complete experiment as a pickle file",
        )

        def prepare_csv_data():
            self.group_manager.sync_all_groups_to_manager()
            df = self.group_manager.bayes_manager.data
            return df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="𝄜 Download as .csv (CANNOT BE REUPLOADED)",
            data=prepare_csv_data(),
            file_name=f"BioKernel_{self.bayes_manager.experiment_name}_{current_time}.csv",
            mime="text/csv",
        )



    def write_footnote(self):
        # Resolve the footnote path relative to this source file. This avoids
        # relying on the current working directory, which can differ (e.g. in
        # Docker or when running from another folder).
        footnote_path = pathlib.Path(__file__).resolve().parent / "footnote.md"

        text = None
        try:
            with open(footnote_path, "r", encoding="utf-8") as f:
                text = f.read()
        except FileNotFoundError:
            # Graceful fallback if the file isn't present in the runtime image.
            text = (
                "**About this app**\n\n"
                "This application displays Bayesian optimization experiments. "
                "(footnote.md not found in the deployment)."
            )
        except Exception as e:
            text = f"Could not load app information: {e}"

        with st.expander("ℹ️ User Instructions", expanded=False):
            st.markdown(text)




if __name__ == "__main__":

    st.set_page_config(
            page_title="BioKernel: Bayesian Optimization for Biological Systems",
            layout="wide",
            initial_sidebar_state="collapsed",
        )


    if "manager" not in st.session_state:
        st.session_state.manager = main_manager()

    st.session_state.manager.main_loop()



