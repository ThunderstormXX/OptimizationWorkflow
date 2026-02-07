# visibility_demo

Test model+optimizer visibility in CSV and plots

> See `artifacts/report.md` for plots and aggregated tables.

## Configuration

- Mode: matrix
- Environment: gossip
- Steps: 10
- Dimension: 5
- Condition number: 10.0
- Optimizer: fw
- Learning rate: 0.1
- Constraint: l2ball
- Radius: 1.0
- Nodes: 4
- Matrix: small
- Seeds: 0
- Save histories: False

## Reproduce

```bash
python -m benchmarks.runner \
    --mode matrix \
    --workflow-dir workflow \
    --env gossip \
    --steps 10 \
    --dim 5 \
    --cond 10.0 \
    --optimizer fw \
    --lr 0.1 \
    --constraint l2ball \
    --radius 1.0 \
    --seeds "0" \
    --matrix small \
    --n-nodes 4 \
    --exp-name "visibility_demo" \
    --description "Test model+optimizer visibility in CSV and plots"
```
