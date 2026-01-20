# big_ablation

Large ablation with plots

> See `artifacts/report.md` for plots and aggregated tables.

## Configuration

- Mode: matrix
- Environment: gossip
- Steps: 30
- Dimension: 10
- Condition number: 10.0
- Optimizer: gd
- Learning rate: 0.1
- Constraint: l2ball
- Radius: 1.0
- Nodes: 5
- Matrix: large
- Seeds: 0,1
- Save histories: False

## Reproduce

```bash
python -m benchmarks.runner \
    --mode matrix \
    --workflow-dir workflow \
    --env gossip \
    --steps 30 \
    --dim 10 \
    --cond 10.0 \
    --optimizer gd \
    --lr 0.1 \
    --constraint l2ball \
    --radius 1.0 \
    --seeds "0,1" \
    --matrix large \
    --n-nodes 5 \
    --exp-name "big_ablation" \
    --description "Large ablation with plots"
```
