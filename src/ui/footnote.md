### About This App
Welcome to BioKernel, the application serves as a no-code enables designing no-code Bayesian Optimization **(BO)** experiments.

BO allows the user to find the optimal parameters for a black-box response function, given a continuous parameter n-dimensional space. Bayesian Optimisation is the the most effective exploration technique and works well in multi-dimensional data.

For more in depth description of why and how BO works, please see the Imperial's iGEM 2025 wiki.

### Quick Start Guide
* **Initialize**: Start a new experiment or load an existing one using the sidebar options.
    * New Experiment allows you to configure:
        * Experiment name and objective direction (maximise or minimise).
        * The Gaussian Process and acquisition function.
            * Note that not all processes work well with all acquisition functions.
            * Certain GP models are designed to work in noiseless settings (e.g. SingleTaskGP), whereas others were specifically built for Heteroscedastic noise (HeteroWhiteSGP). When Signal to Noise ratio is low, technical replicates are encouraged. 
        * Input parameter space.
    * Or load a previously started experiment.
* **Run optimisation**: Create and manage experimental groups, each with its own desirable number of replicates and data.
    * Hitting **Generate New Targets** will allow the BO policy to find you the most salient data points to investigate. The first five data points are generated using SOBOL, before switching to your chosen policy.
    * You can also add groups manually, to allow the experiment their desired freedom.
    * Choose between displaying all and response pending groups for simpler input.
* **Visualize Results**: Utilize the integrated visualization tools to analyze and interpret your experimental results.
    * Best performer plot provide a quick glance at BO progress.
    * Plotting the Gaussian Process **(GP)** at specific coordinates shows a representative slice of the process along each dimension. This is not necessary for BO itself, but is provided as a sanity check and a tool to convince oneself that BO works.

* **Download Data**: Save your experiment state and results, the file can be reuploaded during the next session.
    * If other forms of analysis is desired, the observation data can be downloaded as a .csv file.
    * Ensure to save your experiment data before closing the app to avoid data loss. **There is no auto-save feature**.
    

### Life Cycle
The application was developed to allow easy modification to accommodate the needs of the user. A custom Gaussian Process or an acquisition function can be added in as little as 8 lines of code. The instruction on how to do this can be found on a public repository.

The program was designed to be open-source, by Povilas from Imperial's iGEm 2025 team to support running the lab experiments. 
It will be maintained and updated, however, is static as part of the iGEM competition up until wiki thaw 9th November 2025.
* Main changes will include hosting and data storage solutions, both of which could not be implemented due to iGEM rules for completely static, no external content wiki. However, the file can be manually deployed with just 3 simple commands - please see the appropriate repository to do so. 

For any questions, support running the application, or modification requests please contact me on:
* p.sauciuvienas@gmail.com
* or https://github.com/Sauciu1