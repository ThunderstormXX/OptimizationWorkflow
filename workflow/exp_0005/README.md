# gt_quadratic

Gradient tracking baseline on quadratic

## Configuration

- Mode: single
- Environment: gossip
- Steps: 30
- Dimension: 10
- Condition number: 10.0
- Optimizer: gd
- Learning rate: 0.05
- Constraint: l2ball
- Radius: 1.0
- Seed: 0
- Nodes: 5
- Topology: ring
- Strategy: gradient_tracking

## Reproduce

```bash
python -m benchmarks.runner \
    --mode single \
    --workflow-dir workflow \
    --env gossip \
    --steps 30 \
    --dim 10 \
    --cond 10.0 \
    --optimizer gd \
    --lr 0.05 \
    --constraint l2ball \
    --radius 1.0 \
    --seed 0 \
    --n-nodes 5 \
    --topology ring \
    --strategy gradient_tracking \
    --exp-name "gt_quadratic" \
    --description "Gradient tracking baseline on quadratic"
```
