# Research Project<br>Quantum No-Free-Lunch

## Setup
To set up the project, run <code>poetry install</code>
## Running experiments

The code for generating and running experiments can be found in <code>generate_experiments.py</code>

The parameter configurations for different experiments can be found in <code>config.py</code>, where e.g. <code>get_exp_one_qubit_unitary_config</code> refers to the experimental setup for single-qubit unitary experiment.

To run the single-qubit unitary and the 4-qubit unitary experiments, simply call <code>test_fig2()</code> and <code>test_fig3()</code> to generate the results.

## Dependencies

This repository uses python 3.10, pennylane, torch and qiskit, the specific versions can be found in <code>pyproject.toml</code>
