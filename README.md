# Do-More-with-Less: Coarse-to-Fine Graph Simplification via Dual-Agent

This repository contains the official implementation of our dual‑agent graph simplification algorithm. The core code resides in `Do more with less/highestRL.py`.

---

## Components
This package contains 1 directory

1. Directory `Do more with less` with the source file `highestRL.py` composed of:
	(1). RL agents and NN backbones
		- `one_layer_nn`: a two-layer fully connected Q-network with SELU
		- `BaseAgent`: replay buffer, epsilon-greedy, learning
		- `Agent_C`: Collection agent (coarse), proposes candidate nodes
		- `Agent_R`: Selection agent (fine), picks the best subset
	(2). Action-space constructors
		- `build_collection_action_space_new`: for Agent-C; modes `greedy` / `topk` / `greedy_window`
		- `build_selection_action_space_multi`: for Agent-R; ranks prefix sets by total Laplacian-energy gain
	(3). State-vector builders
		- `build_state_vector_C`: 4 features per candidate (ΔL, Δedges, overlap, clustering)
		- `build_state_vector_R_multi`: 2 features per set (total gain Δ, set size |S|)
	(4). Graph utilities and evaluation
		- `calculate_laplacian_energy`: ℒ(G)=Σ(Dv²)+Σ(Dv) (with caching)
		- `delta_laplacian_energy`, `delta_edges`, `delta_overlap`, `calculate_clustering_coefficient`
		- `evaluate_graph_preservation`: node/edge/energy preservation
		- `plot_graph_comparison`: side-by-side visualization + textual summary
	(5). End-to-end pipelines
		- `highRL`: training loop (optionally runs inference at the end)
		- `infer_highRL`: inference-only pipeline (loads checkpoints or .pth)

---

## Environment Setup

Create and activate a virtual environment:
```bash
python3 -m venv highestRL
source highestRL/bin/activate
# Windows (PowerShell):
# python -m venv highestRL
# .\highestRL\Scripts\Activate.ps1
```

Install dependencies:
```bash
pip install torch
pip install numpy python-igraph psutil matplotlib pandas
```

Deactivate / remove:
```bash
deactivate
rm -rf highestRL
```

Notes
- Uses `python-igraph` (import name `igraph`).
- For GPU, install a CUDA-enabled PyTorch per the official guide.

---

## Datasets

- Use `igraph.Graph` (undirected). Each vertex must have attribute `name` (original id).
- Original energy can be computed by:
```python
calculate_laplacian_energy(G, set(G.vs['name']))
```
- Stopping rule (per connected component): stop once preserved energy ≥ `energy_ratio × original_energy`.

---

## Quick Start

Change into the folder that contains `highestRL.py`:
```bash
cd "Do more with less"
```

### A) Inference demo (build a tiny graph and run)
```python
from highestRL import infer_highRL, calculate_laplacian_energy
import igraph as ig

G = ig.Graph()
G.add_vertices([0,1,2,3,4])
G.vs['name'] = [0,1,2,3,4]
G.add_edges([(0,1),(1,2),(2,3),(3,4),(4,0),(1,3)])

original_energy = calculate_laplacian_energy(G, set(G.vs['name']))
simplified_graph, metrics, inf_time, phase_stats = infer_highRL(
	G_test=G,
	original_energy=original_energy,
	energy_ratio=0.5,
	dataset_name='demo_inference',
	action_space_size=5,
	model_epoch=10000,
	use_checkpoint=False,
	use_amp=True,
	num_workers=8,
	batch_size=256,
	k2=None,
	start_node=None,
	collection_mode='greedy',   # 'greedy' / 'topk' / 'greedy_window'
	window_size=None            # only effective with 'greedy_window'
)

print("Simplified:", simplified_graph.vcount(), "nodes,", simplified_graph.ecount(), "edges")
print("Metrics:", metrics)
print("Inference time:", inf_time, "seconds")
```

### B) Training + inference (minimal)
```python
from highestRL import highRL, calculate_laplacian_energy
import igraph as ig

G = ig.Graph()
G.add_vertices([0,1,2,3,4])
G.vs['name'] = [0,1,2,3,4]
G.add_edges([(0,1),(1,2),(2,3),(3,4),(4,0),(1,3)])

original_energy = calculate_laplacian_energy(G, set(G.vs['name']))
agent_C, agent_R, simplified_graph, metrics, inf_time, phase_stats = highRL(
	G_train=G,
	original_energy=original_energy,
	energy_ratio=0.5,
	dataset_name='demo_train',
	train=True,
	training_epoch=100,        # reduce for demo
	action_space_size=5,
	alpha=0.003,
	save_model_interval=100,
	target_update=100,
	run_inference=True,
	resume_checkpoint=None,
	checkpoint_dir='checkpoints',
	use_amp=True,
	num_workers=0,
	batch_size=64,
	k2=None,
	test_interval=1000,
	num_test_runs=0,
	G_test=None,
	test_original_energy=None,
	start_epoch=0,
	train_start_node=None,
	collection_mode='greedy',
	window_size=None
)

print("Training+Inference done. Final metrics:", metrics)
```

---

## CLI Usage (only required parameters)

We expose only the four essential inputs; all other arguments use defaults.

Important: `window_size` takes effect only when `--collection_mode greedy_window` is set. Otherwise the default mode is `greedy` and `window_size` is ignored.

- Train
```bash
python highestRL.py \
  --mode train \
  --dataset_name <DATASET_NAME> \
  --energy_ratio <RATIO> \
  --training_epoch <EPOCHS> \
  --collection_mode greedy_window \
  --window_size <WINDOW>
```

- Inference
```bash
python highestRL.py \
  --mode infer \
  --dataset_name <DATASET_NAME> \
  --energy_ratio <RATIO> \
  --collection_mode greedy_window \
  --window_size <WINDOW>
```

Parameter notes (this section lists only the essentials):
- `--dataset_name`: Dataset name (used for logging/start-node hints; not for file I/O).
- `--energy_ratio`: Target energy preservation ratio (e.g., `0.5` for 50%).
- `--training_epoch`: Number of training epochs (train mode only).
- `--collection_mode`: Use `greedy_window` to enable windowing; otherwise default is `greedy`.
- `--window_size`: Window size (effective only with `greedy_window` and > 0).


---

