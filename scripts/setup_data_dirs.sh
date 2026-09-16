#!/bin/bash
# Helper script to set up the standard data directory structure

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${PROJECT_ROOT}/data"

echo "Setting up standard data directory structure at: $DATA_DIR"

# Create subdirectories
mkdir -p "${DATA_DIR}/datasets"
mkdir -p "${DATA_DIR}/partitions"
mkdir -p "${PROJECT_ROOT}/logs"

echo "✓ Created directories:"
echo "  - ${DATA_DIR}/datasets"
echo "  - ${DATA_DIR}/partitions"
echo "  - ${PROJECT_ROOT}/logs"

echo ""
echo "Next steps:"
echo "1. Copy your custom graph data to: ${DATA_DIR}/datasets/my_custom_graph/"
echo "   Required files:"
echo "     - node_embeddings_v.pt (shape: num_nodes × 512)"
echo "     - labels.pt (shape: num_nodes)"
echo "     - edge_list or CSR format file"
echo ""
echo "2. Update slurm/example_config.sh:"
echo "     DATA_DIR=\"\${PROJ_PATH}/data/datasets\""
echo "     PARTITION_DIR=\"\${PROJ_PATH}/data/partitions\""
echo "     DATASET_NAME=\"my_custom_graph\""
echo ""
echo "3. Partition the graph:"
echo "     cd partition"
echo "     sbatch partition.sh my_custom_graph metis \"4\" ../data/datasets ../data/partitions"
echo ""
echo "4. Run training:"
echo "     cd slurm"
echo "     bash set_params.sh --config example_config.sh"
