# Overview

Enhanced Adaptive DBSCAN extends DBSCAN with adaptive parameters and stability analysis, making it robust on datasets with varying densities (e.g. wafer defect maps).

## Key ideas
- Local-density–aware epsilon and MinPts
- Stability-based cluster selection across scales
- Incremental and partial reclustering
- Boundary-aware density adjustment for wafer shapes

## Architecture at a glance

```mermaid
flowchart TD
  A[Input Data X + optional features] --> B[Preprocess / scale features]
  B --> C[Local density estimation kNN / KDTree]
  C --> D[Adaptive parameter computation epsilon, MinPts]
  D --> E[Core point identification]
  E --> F[Cluster formation region expansion]
  F --> G[Hierarchy build & stability scoring]
  G --> H[Stable cluster selection]
  H --> I[Assign full data & update centers]
  I --> J[Metrics & plotting]
```
