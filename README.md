# Intro 
Welcome to BioKernel, a software tool developed for no-code Bayesian optimisation of biological systems to support Imperial's iGEM 2025. For a more in depth description of why or how it runs please refer the Imperial iGEM 2025 wiki.

There are two main parts to this repository:
* The BioKernel software, which can be run for simple shell commands and accessed from within your browser. It is self-encompassing and requires no further interactions with the repository. 

* The supporting simulations and notebooks for the custom GP kernel developed for technical repeats and heteroscedastic noise can be found within the ./notebooks/ folder.


# Software
The package provides a no-code solution for running Bayesian Experiments. All instructions and description can be found within the applet itself and the accompanying iGEM pages.

## Local Deployment

To deploy BioKernel, you first need [docker](https://www.docker.com/) and [git](https://git-scm.com/downloads) installed on your system. Please don't forget to start docker.

``` bash
git clone https://gitlab.igem.org/2025/software-tools/imperial
cd imperial
docker build -t BioKernel:v11 .
```

This will take ~10 minutes to compile, afterwards it can be rapidly deployed and accessed from your browser after starting docker and running the image.

``` bash
docker run -p 8989:8989 BioKernel:v11
```

The link to the applet for your browser is then gonna be:
``` bash
http://localhost:8989/
```

## Addition of Custom Gaussian Processes and Acquisition Functions

To support the users needs, custom gaussian process and acquisition functions (BOTorch and GPyTorch implementations) can easily be included in the UI.

To do so you just need to add your desired custom class to the[src/gp_and_acq_f/custom_gp_and_acq_f.py](./src/gp_and_acq_f/custom_gp_and_acq_f.py) file.

Simplest example is just the wrapper for SingleTaskGP:

``` python 
from botorch.models import SingleTaskGP

class bocoli_SingleTaskGP(SingleTaskGP):
    """A SingleTaskGP wrapper that works with botorch"""
    bocoli_name = "SingleTaskGP"
    bocoli_type = "gp"
    bocoli_description = """Standard Single Task Gaussian Process from BoTorch.
    Cannot handle noise, fits through every observed point"""

    def __init__(self, train_X, train_Y, **kwargs):
        super().__init__(train_X, train_Y, **kwargs)
```

For the BioKernel loader to discover your custom function it needs to have two attributes:
* `bocoli_name` - the name by which your class will be displayed in the user UI.
* `bocoli_type` - describes the type of class
    * `gp` indicates a gaussian process.
    * `acqf` indicates an acquisition function.
* `bocoli_description` is not mandatory, but makes tracking and choosing functions in the UI a bit simpler. 

Afterwards, the app needs to be rerun (or docker image rebuilt). Please see the instructions [bellow](#running-the-software-without-docker).

# Simulations and the notebooks


The analysis and extrapolation models used to generate the BO simulation run can be found at:
[notebooks/confirmation_on_empyrical_data.ipynb](notebooks/confirmation_on_empyrical_data.ipynb)

To run the project locally you will need python 3.11 installed on the system, from within project folder:


``` bash
pip install poetry
poetry lock
poetry install
poetry env activate
```

### Running the software without docker
After installing the environment you:

``` bash
poetry run streamlit run ./src/ui/main_ui.py --server.port=8989 --server.address=0.0.0.0
```
BioKernel will once again be accessible via browser at:

``` bash
http://localhost:8989/
```



# References and attributions:

### Data Source

The experimental data [data/melavonate_pathway/ao2c00483_si_002.xlsx](data/melavonate_pathway/ao2c00483_si_002.xlsx) used to confirm success of our BO was taken from:

``` text
@article{Shin2022,
  title = {Transcriptional Tuning of Mevalonate Pathway Enzymes to Identify the Impact on Limonene Production in Escherichia coli},
  author = {Shin, Jonghyeon and South, Eric J. and Dunlop, Mary J.},
  journal = {ACS Omega},
  year = {2022},
  volume = {7},
  number = {22},
  pages = {18331--18338},
  publisher = {American Chemical Society},
  doi = {10.1021/acsomega.2c00483},
  url = {https://doi.org/10.1021/acsomega.2c00483}
}
```


### Libraries Used:

```text
@inproceedings{olson2025ax,
  title = {{Ax: A Platform for Adaptive Experimentation}},
  author = {
    Olson, Miles and Santorella, Elizabeth and Tiao, Louis C. and
    Cakmak, Sait and Garrard, Mia and Daulton, Samuel and
    Lin, Zhiyuan Jerry  and Ament, Sebastian and Beckerman, Bernard and
    Onofrey, Eric and Igusti, Paschal and Lara, Cristian and
    Letham, Benjamin and Cardoso, Cesar and Shen, Shiyun Sunny and
    Lin, Andy Chenyuan and Grange, Matthew and Kashtelyan, Elena and
    Eriksson, David and Balandat, Maximilian and Bakshy, Eytan.
  },
  booktitle = {AutoML 2025 ABCD Track},
  year = {2025}
}
```

```text
@inproceedings{balandat2020botorch,
  title = {{BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization}},
  author = {Balandat, Maximilian and Karrer, Brian and Jiang, Daniel R. and Daulton, Samuel and Letham, Benjamin and Wilson, Andrew Gordon and Bakshy, Eytan},
  booktitle = {Advances in Neural Information Processing Systems 33},
  year = 2020,
  url = {http://arxiv.org/abs/1910.06403}
}
```
