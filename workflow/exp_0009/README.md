# anim_demo

Gossip logistic with animated visualization

> See `artifacts/report.md` for plots and aggregated tables.

## Configuration

- Mode: single
- Environment: gossip
- Steps: 60
- Dimension: 10
- Condition number: 10.0
- Optimizer: gd
- Learning rate: 0.1
- Constraint: l2ball
- Radius: 1.0
- Seed: 0
- Nodes: 8
- Topology: ring
- Strategy: local_then_gossip

## Reproduce

```bash
python -m benchmarks.runner \
    --mode single \
    --workflow-dir workflow \
    --env gossip \
    --steps 60 \
    --dim 10 \
    --cond 10.0 \
    --optimizer gd \
    --lr 0.1 \
    --constraint l2ball \
    --radius 1.0 \
    --seed 0 \
    --n-nodes 8 \
    --topology ring \
    --strategy local_then_gossip \
    --exp-name "anim_demo" \
    --description "Gossip logistic with animated visualization"
```
