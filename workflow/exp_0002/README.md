# optim_registry_demo

Compare fw vs pgd across strategies/topologies

## Configuration

- Mode: matrix
- Environment: gossip
- Steps: 20
- Dimension: 10
- Condition number: 10.0
- Optimizer: fw
- Learning rate: 0.05
- Constraint: l2ball
- Radius: 2.0
- Nodes: 5
- Matrix: small
- Seeds: 0
- Save histories: False

## Reproduce

```bash
python -m benchmarks.runner \
    --mode matrix \
    --workflow-dir workflow \
    --env gossip \
    --steps 20 \
    --dim 10 \
    --cond 10.0 \
    --optimizer fw \
    --lr 0.05 \
    --constraint l2ball \
    --radius 2.0 \
    --seeds "0" \
    --matrix small \
    --n-nodes 5 \
    --exp-name "optim_registry_demo" \
    --description "Compare fw vs pgd across strategies/topologies"
```
