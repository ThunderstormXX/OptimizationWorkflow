# mnist_gossip_fixture_test

> See `artifacts/report.md` for plots and aggregated tables.

## Configuration

- Mode: single
- Environment: gossip
- Steps: 3
- Dimension: 10
- Condition number: 10.0
- Optimizer: adam
- Learning rate: 0.01
- Constraint: none
- Radius: 1.0
- Seed: 0
- Nodes: 2
- Topology: ring
- Strategy: local_then_gossip

## Reproduce

```bash
python -m benchmarks.runner \
    --mode single \
    --workflow-dir workflow \
    --env gossip \
    --steps 3 \
    --dim 10 \
    --cond 10.0 \
    --optimizer adam \
    --lr 0.01 \
    --constraint none \
    --radius 1.0 \
    --seed 0 \
    --n-nodes 2 \
    --topology ring \
    --strategy local_then_gossip \
    --exp-name "mnist_gossip_fixture_test"
```
