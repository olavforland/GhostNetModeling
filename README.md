# GhostNetModeling
Repository for the project in the course TTK4854 - Robotic Ocean Waste Removal

## General info

The repository consists of two main folders: `LobsterTrapModeling` and `FishingNetSimulation`. They contain two different modeling pipelines: Modeling of lobster traps, which are assumed to not move around when dropped in the ocean, and modeling of lost fishing nets, which are assumed to move with the ocean currents. 

## Setup

To get started, navigate to the project folder and run the following commands. The creating and activating of a conda environment is not needed, but recommended. 

```
conda create –name <env_name> python=3.11 
conda activate <env_name>
pip install -r requirements.txt
```

## Lobster trap modeling

For more in-depth explanation of how the classes interact, see `/LobsterTrapModeling/modeling_walkthrough.ipynb`

The most important classes are found in `location_model.py`:
- `LobsterTrapLocationModel`: Wrapper class for `sklearn.mixture.GaussianMixture`. Contains a pd.DataFrame of coordinates, and handles fitting the mixture model to the data. Intended to fit data at a specific location, as it gets inaccurate when the area the points span gets to large.
- `RegionalLobsterTrapModel`: Contains all data points for Norway, and handles clustering the data and fitting a `LobsterTrapLocationModel` to each of the clusters. 
- `RegionalFoundTrapsLocationModel`: Interface for modeling the **found** lobster traps. Inherits from `RegionalLobsterTrapModel`
- `RegionalLostTrapsLocationModel`: Interface for modeling the **lost** lobster traps. Inherits from `RegionalLobsterTrapModel`
