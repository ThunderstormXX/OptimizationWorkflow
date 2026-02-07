# exp_0031

> See `artifacts/report.md` for plots and aggregated tables.

## Configuration

- Mode: single
- Environment: single
- Steps: 50
- Dimension: 10
- Condition number: 10.0
- Optimizer: fw
- Learning rate: 0.1
- Constraint: l2ball
- Radius: 1.0
- Seed: 0
- Step schedule: harmonic

## Reproduce

```bash
python -m benchmarks.runner \
    --mode single \
    --workflow-dir workflow \
    --env single \
    --steps 50 \
    --dim 10 \
    --cond 10.0 \
    --optimizer fw \
    --lr 0.1 \
    --constraint l2ball \
    --radius 1.0 \
    --seed 0 \
    --step-schedule harmonic
```
