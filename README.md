# Data Driven Optimal Investment Using Neural Networks

### This code is part of a thesis submission for the MSc Mathematical Finance Programme.

## Installing and running
* Install requirements via: pip install -r requirements.txt
* To run this code:
  * Pick a configuration JSON from the config folder (file.json)
  * Execute: python run_experiment.py file.json

## Modules
The wealth_optimizer package contains the following:
* bootstrap.py - implementation of stationary and circular bootstrap algorithms
* common_logger.py - logging utility
* data_loader.py - Excel file reader utility
* data_slicing.py - utility for splitting a time series dataset into training and test data
* executors.py - houses the model executor code which handles the training of the NN-based control model
* loss_functions.py - implementations of loss functions for the neural network
* models.py - implementations of the leveraged long only and long/short neural control models
* optimal_window_selection.py - implementation of the Politis & White optimal bootstrap block selection
* performance_stats.py - utilities for analysing and plotting results
* simulated_bm.py - utilities for generating Geometric Brownian Motions for testing purposes
* utils.py - general utilities
