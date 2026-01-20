# matrix_demo

Compare gossip strategies/schedules/topologies

## Configuration

- Mode: matrix
- Environment: gossip
- Steps: 20
- Dimension: 10
- Condition number: 10.0
- Constraint: l2ball
- Radius: 2.0
- Nodes: 5
- Matrix: small
- Seeds: 0,1
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
    --constraint l2ball \
    --radius 2.0 \
    --seeds "0,1" \
    --matrix small \
    --n-nodes 5 \
    --exp-name "matrix_demo" \
    --description "Compare gossip strategies/schedules/topologies"
```
