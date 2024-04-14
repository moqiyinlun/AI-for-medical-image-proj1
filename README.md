# AI-for-medical-image-proj1
The processed cine.npz can be found in http://10.19.136.64:5000/sharing/EjSxgyWgR use campus internet, you should put it into "data/" folder.

## Description

This project primarily adapts the widely-used CNN networks from static MRI reconstruction for dynamic MRI reconstruction. In addition, to enhance the reconstruction quality of edges and areas with significant grayscale variations, we have incorporated Edge Loss and Perceptual Loss to improve the network's performance in these regions. Furthermore, the concept of cascading has been introduced to further elevate network performance.

## Installation

To set up the project environment:

1. Download the repository to your local machine.
2. Ensure that you have Python installed, preferably version 3.8 or above.

## Usage

The project is structured into different Jupyter Notebooks, each corresponding to a specific problem or visualization aspect:

- `problem1.ipynb` - `problem5.ipynb`: Each notebook contains a different problem and its solution.
- `problem2_additional_loss.ipynb`: Model with edge loss and Perceptual loss.
- `visualization_2.ipynb` - `visualization_5.ipynb`: These notebooks are for visualizing the results of the experiments.
- `calculate_for_train.ipynb`: Utils to load model and visualize.

To view and run the notebooks, use the following command:

```bash
jupyter notebook <notebook_name>.ipynb
```

## Utils
loss_util.py: Utility functions for loss calculations.
resblock.py: Residual block implementation.
show.py: Functions to display images.