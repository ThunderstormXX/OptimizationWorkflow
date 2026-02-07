# mnist_anim_demo

MNIST animation demo

> See `artifacts/report.md` for plots and aggregated tables.

## Configuration

- Mode: single
- Environment: single
- Steps: 10
- Dimension: 10
- Condition number: 10.0
- Optimizer: adam
- Learning rate: 0.01
- Constraint: none
- Radius: 1.0
- Seed: 0

## Reproduce

```bash
python -m benchmarks.runner \
    --mode single \
    --workflow-dir workflow \
    --env single \
    --steps 10 \
    --dim 10 \
    --cond 10.0 \
    --optimizer adam \
    --lr 0.01 \
    --constraint none \
    --radius 1.0 \
    --seed 0 \
    --exp-name "mnist_anim_demo" \
    --description "MNIST animation demo"
```
