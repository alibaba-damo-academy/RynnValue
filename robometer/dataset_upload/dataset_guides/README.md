# Dataset Guides Overview

This directory contains detailed guides for specific datasets supported by the Robometer training pipeline.

## Available Datasets

### Production-Ready
| Dataset | Type | Size | Features | Guide |
|---------|------|------|----------|-------|
| **AgiBotWorld** | Real Robot | 600GB+ | Streaming, Head Camera | [📖 AgiBotWorld.md](AgiBotWorld.md) |
| **LIBERO** | Simulation | ~5GB | HDF5, Multi-env | [📖 LIBERO.md](LIBERO.md) |
| **FinoNet** | Manipulation | ~10GB | Success/Failure, PNG | [📖 FinoNet.md](FinoNet.md) |
| **H2R** | Human-Robot | ~ | Paired videos | [📖 H2R.md](H2R.md) |

### Custom Integration
| Type | Description | Guide |
|------|-------------|-------|
| **Custom Dataset** | Add DROID, Bridge, or any dataset | [📖 CustomDataset.md](CustomDataset.md) |

## Quick Reference

### AgiBotWorld (Streaming)
```bash
uv run python data/generate_hf_dataset.py --config_path=configs/data_gen_configs/agibot_world.yaml
```
- ✅ **No download needed** (600GB+ dataset streams)
- ✅ **Authentication required** (HuggingFace login + license)
- ✅ **Head camera only** (ignores depth/other views)

### LIBERO (Local Files)  
```bash
uv run python data/generate_hf_dataset.py --config_path=configs/data_gen.yaml
```
- ✅ **Local HDF5 files** (download LIBERO dataset first)
- ✅ **Multi-environment** (living room, kitchen, office, study)
- ✅ **Simulation data** (high-quality manipulation tasks)

### FinoNet (Success/Failure Data)
```bash
uv run python dataset_upload/generate_hf_dataset.py --config_path=dataset_upload/configs/data_gen_configs/fino_net.yaml
```
- ✅ **PNG sequences** (download from HuggingFace)
- ✅ **5 manipulation tasks** (put_on, put_in, place, pour, push)
- ✅ **Labeled success/failure** episodes for robust training

### Custom Dataset
```bash
# 1. Create loader: data/{name}_loader.py
# 2. Add to converter: data/generate_hf_dataset.py  
# 3. Create config: configs/data_gen_configs/{name}.yaml
# 4. Test: uv run python data/generate_hf_dataset.py --config_path=...
```
- ✅ **Template provided** (follow CustomDataset.md)
- ✅ **Multiple formats** (HDF5, JSON, pickle, etc.)
- ✅ **Flexible integration** (video files, frame arrays, or streaming)

## Directory Structure

```
data/dataset_guides/
├── README.md              ← This overview
├── AgiBotWorld.md         ← Streaming dataset guide  
├── LIBERO.md              ← Local HDF5 dataset guide
├── FinoNet.md             ← Success/failure dataset guide
├── H2R.md                 ← Human-robot paired dataset guide
└── CustomDataset.md       ← Template for new datasets
```

## Getting Started

1. **Choose your dataset** from the table above
2. **Read the specific guide** for detailed instructions
3. **Follow prerequisites** (authentication, downloads, etc.)
4. **Run the conversion** using provided commands
5. **Integrate with training** using the generated dataset

## Need Help?

- 📖 **Main Guide**: [../README_ADDING_DATASETS.md](../README_ADDING_DATASETS.md)
- 🐛 **Issues**: Check troubleshooting sections in individual guides
- 💡 **Custom Datasets**: Start with [CustomDataset.md](CustomDataset.md) template

## Contributing

To add a new dataset guide:

1. Create `{DatasetName}.md` in this directory
2. Follow the structure of existing guides
3. Update this README's table
4. Add reference in main README_ADDING_DATASETS.md