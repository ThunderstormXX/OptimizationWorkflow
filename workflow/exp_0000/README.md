# demo

FW + gossip on synthetic quadratic

## Configuration

- Environment: gossip
- Steps: 20
- Seed: 0
- Dimension: 10
- Condition number: 10.0
- Constraint: l2ball
- Step schedule: harmonic
- Nodes: 5
- Topology: ring

## Reproduce

```bash
python -m benchmarks.runner \
    --workflow-dir workflow \
    --env gossip \
    --steps 20 \
    --seed 0 \
    --dim 10 \
    --cond 10.0 \
    --constraint l2ball \
    --step-schedule harmonic \
    --radius 1.0 \
    --n-nodes 5 \
    --topology ring \
    --exp-name "demo" \
    --description "FW + gossip on synthetic quadratic"
```
