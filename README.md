# Research Project<br>Quantum No-Free-Lunch

This package uses Pennylane.


## To do:
- Optimize number of reference qbits (r_qbits  $=\lceil log_2 r\rceil$, where $r$ is max. Schmidt rank in the training set)
- Test different learning rates
- Test different termination conditions (max steps, low cost,...)

## Experiments
- Reproduce Figure 2 and Figure 3:
  - Fig. 2: 1 qubit, 10 unitaries and 10 training sets with entangled/unentangled states
  - Fig. 3: 6 qubits, 10 unitaries, each trained on 100 training sets with the same Schmidt rank $r = 2^0,\dots, 2^6$
- Training sets with non-uniform Schmidt rank
## Meeting 16.6.:
- Server for running performance heavy experiments?