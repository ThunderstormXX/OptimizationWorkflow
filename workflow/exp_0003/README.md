# checks_demo

Regression checks: FW gossip ring

## Configuration

- Mode: checks
- Environment: gossip
- Steps: 30
- Dimension: 10
- Condition number: 10.0
- Optimizer: fw
- Learning rate: 0.1
- Constraint: l2ball
- Radius: 2.0
- Seed: 0
- Step schedule: harmonic
- Nodes: 5
- Topology: ring
- Strategy: local_then_gossip

## Reproduce

```bash
python -m benchmarks.runner \
    --mode checks \
    --workflow-dir workflow \
    --env gossip \
    --steps 30 \
    --dim 10 \
    --cond 10.0 \
    --optimizer fw \
    --lr 0.1 \
    --constraint l2ball \
    --radius 2.0 \
    --seed 0 \
    --step-schedule harmonic \
    --n-nodes 5 \
    --topology ring \
    --strategy local_then_gossip \
    --exp-name "checks_demo" \
    --description "Regression checks: FW gossip ring"
```
