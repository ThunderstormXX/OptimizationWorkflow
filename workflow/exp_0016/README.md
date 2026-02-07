# mnist_gossip_demo

MNIST gossip demo with FakeData

> See `artifacts/report.md` for plots and aggregated tables.

## Configuration

- Mode: single
- Environment: gossip
- Steps: 5
- Dimension: 10
- Condition number: 10.0
- Optimizer: adam
- Learning rate: 0.01
- Constraint: none
- Radius: 1.0
- Seed: 0
- Nodes: 3
- Topology: ring
- Strategy: local_then_gossip

## Reproduce

```bash
python -m benchmarks.runner \
    --mode single \
    --workflow-dir workflow \
    --env gossip \
    --steps 5 \
    --dim 10 \
    --cond 10.0 \
    --optimizer adam \
    --lr 0.01 \
    --constraint none \
    --radius 1.0 \
    --seed 0 \
    --n-nodes 3 \
    --topology ring \
    --strategy local_then_gossip \
    --exp-name "mnist_gossip_demo" \
    --description "MNIST gossip demo with FakeData"
```
