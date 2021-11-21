# MLPMixers Study
Comparing MLPMixers to vanilla MLP and convolution-based vision algorithms.

## Setting Up Project via a Traditional Installation (for Linux-Based Operating Systems)

First, install and configure Conda environment:

```bash
# Clone this repository:
git clone https://github.com/amorehead/MLPMixers-Study

# Change to project directory:
cd MLPMixers-Study

# Set up Conda environment locally
conda env create --name mlpmixers_study -f environment.yml

# Activate Conda environment located in the current directory:
conda activate mlpmixers_study

# (Optional) Perform a full install of the pip dependencies described in 'requirements.txt':
pip3 install -r requirements.txt

# (Optional) To remove the long Conda environment prefix in your shell prompt, modify the env_prompt setting in your .condarc file with:
conda config --set env_prompt '({name})'
 ```

## Running Project after Performing a Traditional Installation (for Linux-Based Operating Systems)

Run like a typical Python script:

```bash
# Run the PyTorch Lightning script:
python3 lit_mlp_mixer.py
 ```