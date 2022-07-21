# Research Project<br>Quantum No-Free-Lunch

This package uses Pennylane.

## Momentane Erxperiment:
1. 25-50 layer für alle netze, für max. schmidt rank auf 64 samples
2. 1-25 layer für alle netze, für max. schimdt rank auf 64 samples
3. 25-50 layer für alle netze, für max. schmidt rank auf 64 samples und ReduceLRonPlateau

##Ideen:
- weight intilization ändern
- mehr Circuits testen
- Dynamic Learning Rate 

## To do:
- Test different qnns
- Test different learning rates
- Test different termination conditions (max steps, low cost,...)
- 2 qubit unitary expirement on real quantum computer
- Heatmap for 2 variable parameters (all others fixed) and loss as heatmap value

## Experiments
- Reproduce Figure 2 and Figure 3:
  - Fig. 2: 1 qubit, 10 unitaries and 10 training sets with entangled/unentangled states
  - Fig. 3: 6 qubits, 10 unitaries, each trained on 100 training sets with the same Schmidt rank $r = 2^0,\dots, 2^6$
- Training sets with non-uniform Schmidt rank
