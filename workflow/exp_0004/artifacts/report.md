# Matrix Experiment Report

## Overview

- **Number of runs**: 12
- **Artifacts directory**: `workflow/exp_0004/artifacts/runs`

## Best Runs

### Best by Final Suboptimality

- **Run ID**: `run_0009`
- **Optimizer**: pgd
- **Topology**: ring
- **Strategy**: gossip_then_local
- **Step schedule**: na
- **Seed**: 0
- **Final suboptimality**: 0.001999

### Best by Final Mean Loss

- **Run ID**: `run_0010`
- **Optimizer**: pgd
- **Topology**: complete
- **Strategy**: local_then_gossip
- **Step schedule**: na
- **Seed**: 0
- **Final mean loss**: -1.536701

## Top 5 Runs by Final Suboptimality

| Rank | Run ID | Optimizer | Topology | Strategy | Schedule | Seed | Suboptimality |
|------|--------|-----------|----------|----------|----------|------|---------------|
| 1 | `run_0009` | pgd | ring | gossip_then_local | na | 0 | 0.001999 |
| 2 | `run_0010` | pgd | complete | local_then_gossip | na | 0 | 0.001999 |
| 3 | `run_0011` | pgd | complete | gossip_then_local | na | 0 | 0.001999 |
| 4 | `run_0008` | pgd | ring | local_then_gossip | na | 0 | 0.001999 |
| 5 | `run_0006` | fw | complete | gossip_then_local | harmonic | 0 | 0.110316 |

## Averages

- **Mean final suboptimality**: 0.174950
- **Mean final mean loss**: -1.363750
- **Mean final consensus error**: 0.000000

## Means by Optimizer

| Optimizer | Runs | Mean Suboptimality | Mean Consensus Error |
|-----------|------|--------------------|-----------------------|
| fw | 8 | 0.261425 | 0.000000 |
| pgd | 4 | 0.001999 | 0.000000 |

## Means by Topology

| Topology | Runs | Mean Suboptimality | Mean Consensus Error |
|----------|------|--------------------|-----------------------|
| complete | 6 | 0.165654 | 0.000000 |
| ring | 6 | 0.184245 | 0.000001 |

## Means by Strategy

| Strategy | Runs | Mean Suboptimality | Mean Consensus Error |
|----------|------|--------------------|-----------------------|
| gossip_then_local | 6 | 0.165758 | 0.000001 |
| local_then_gossip | 6 | 0.184141 | 0.000000 |
