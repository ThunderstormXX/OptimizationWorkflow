# final_visibility_test

Final test: model+optimizer visible

> See `artifacts/report.md` for plots and aggregated tables.

## Configuration

- Mode: single
- Environment: gossip
- Steps: 20
- Dimension: 5
- Condition number: 10.0
- Optimizer: gd
- Learning rate: 0.1
- Constraint: l2ball
- Radius: 1.0
- Seed: 42
- Nodes: 5
- Topology: ring
- Strategy: local_then_gossip

## Reproduce

```bash
python -m benchmarks.runner \
    --mode single \
    --workflow-dir workflow \
    --env gossip \
    --steps 20 \
    --dim 5 \
    --cond 10.0 \
    --optimizer gd \
    --lr 0.1 \
    --constraint l2ball \
    --radius 1.0 \
    --seed 42 \
    --n-nodes 5 \
    --topology ring \
    --strategy local_then_gossip \
    --exp-name "final_visibility_test" \
    --description "Final test: model+optimizer visible"
```
