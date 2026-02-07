# Experiment Report

## Configuration

- **Environment**: gossip
- **Task**: mnist
- **Optimizer**: adam
- **Steps**: 3
- **Seed**: 0
- **Dimension**: N/A
- **Nodes**: 2
- **Topology**: ring
- **Strategy**: local_then_gossip

## Final Metrics

- **Final mean loss**: 2.307701
- **Final mean accuracy**: 0.1562
- **Final consensus error**: 0.000000

## Plots

### Loss vs Step

![Loss](plots/loss.png)

### Consensus Error vs Step

![Consensus](plots/consensus.png)
