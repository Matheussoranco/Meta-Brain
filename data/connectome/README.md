# MaleCNS Connectome Data

This directory stores the *Drosophila* male CNS connectome data files.

## Download Instructions

### Option 1: Automated Download (Recommended)

```bash
# From project root
python scripts/download_data.py --data-dir data/connectome --version v1.0 --files all
```

Requires `gsutil` (Google Cloud SDK) for fastest downloads:
```bash
# Install Google Cloud SDK
# Windows: https://cloud.google.com/sdk/docs/install
# Then: gcloud auth login
```

### Option 2: Manual Download

Visit the official download page: **https://male-cns.janelia.org/download/**

Key files for Meta-Brain:

| File | Description | Size | Required |
|------|-------------|------|----------|
| `malecns_v1_0_meta.feather` | Neuron metadata (type, position, NT, etc.) | ~10 MB | **Yes** |
| `malecns_v1_0_simple_edgelist.feather` | Neuron-to-neuron connectivity | ~3.2 GB | **Yes** |
| `malecns_v1_0_split_edgelist.feather` | Compartment-level connectivity | ~4.6 GB | Optional |
| `malecns_v1_0_synapses.parquet` | Individual synapse locations | ~7.9 GB | Optional |
| `malecns_malecns_space_swc/` | Skeletons in native space | ~2 GB | Optional |
| `malecns_banc_space_swc/` | Skeletons in BANC space | ~2 GB | Optional |
| `obj/neuropils/` | Neuropil meshes | ~500 MB | Optional |

### Option 3: Programmatic Access (neuPrint)

No download needed - query directly from Python:

```python
from neuprint import Client
client = Client('https://neuprint.janelia.org', dataset='male-cns:v1.0', token=YOUR_TOKEN)
```

Get token from: https://neuprint.janelia.org → Account → Token

## Expected Directory Structure

```
data/connectome/
├── malecns_v1_0_meta.feather
├── malecns_v1_0_simple_edgelist.feather
├── malecns_v1_0_split_edgelist.feather (optional)
├── malecns_v1_0_synapses.parquet (optional)
├── malecns_malecns_space_swc/ (optional)
│   └── *.swc
├── malecns_banc_space_swc/ (optional)
│   └── *.swc
└── obj/
    └── neuropils/
        └── *.obj
```

## Data Schema

### Neuron Metadata (`*_meta.feather`)

| Column | Description |
|--------|-------------|
| `bodyId` | Unique neuron identifier |
| `type` | Cell type name |
| `instance` | Left/right instance |
| `superclass` | Major class (OLSN, ORN, PN, MN, DN, etc.) |
| `class` | Subclass |
| `subtype` | Fine type |
| `somaSide` | Left/Right |
| `rootSide` | Left/Right |
| `itoleeHl` | Hemilineage (Lee/Ito) |
| `trumanHl` | Hemilineage (Truman) |
| `flow` | Information flow layer (sensory→motor) |
| `predictedNt` | Neurotransmitter (ACh, GABA, Glu, DA, OA, 5HT) |
| `flywireType` | Matched FlyWire type |
| `mancType` | Matched MANC type |
| `hemibrainType` | Matched hemibrain type |
| `fru` | fruitless expression (0/1) |
| `dsx` | doublesex expression (0/1) |
| `x, y, z` | Soma position (nm) |

### Edgelist (`*_simple_edgelist.feather`)

| Column | Description |
|--------|-------------|
| `bodyId_pre` | Presynaptic neuron ID |
| `bodyId_post` | Postsynaptic neuron ID |
| `weight` | Synapse count |
| `roi` | Neuropil region |

## Quick Test

After downloading, verify the data loads:

```python
from connectome.loader import create_default_loader

loader = create_default_loader()
graph = loader.load_from_files()
print(f"Loaded {len(graph.neurons)} neurons, {len(graph.edges)} connections")

# Get sensorimotor subgraph
sensorimotor = loader.get_sensorimotor_subgraph(max_interneuron_layers=3)
print(f"Sensorimotor: {len(sensorimotor.neurons)} neurons")
```

## Notes

- **v1.0** (released 2026-06-08) is the final release
- **v0.9** (2025-10-03) is the pre-release, still usable but v1.0 preferred
- Data is licensed **CC-BY** — cite Berg et al. (2026) Cell
- Total download for core files: **~3.5 GB**
- Full dataset with all options: **~20 GB**

## Troubleshooting

### gsutil permission denied
```bash
gcloud auth login
gcloud auth application-default login
```

### Feather/Parquet read errors
```bash
pip install --upgrade pyarrow pandas
```

### Memory issues loading edgelist
Use chunked reading or filter to sensorimotor subgraph first:
```python
# In loader.py, modify load_from_files to read in chunks
# Or use sensorimotor subgraph extraction which is smaller
```