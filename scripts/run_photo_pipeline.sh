#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Photo Graph Processing Pipeline"
echo "=========================================="
echo ""

# Verify imports and run pipeline via conda
echo "1️⃣  Verifying environment..."
conda run -n llm-dgl-cpu python3 -c "
import torch, dgl, numpy, scipy, pandas
print(f'✓ PyTorch {torch.__version__}')
print(f'✓ DGL {dgl.__version__}')
print(f'✓ NumPy {numpy.__version__}')
print(f'✓ SciPy {scipy.__version__}')
"
echo ""

# Partition the graph
echo "2️⃣  Partitioning photo graph (4 partitions)..."
cd partition
conda run -n llm-dgl-cpu python3 partition_graph.py \
    --dataset photo \
    --dataset_dir ../data/photo \
    --num_parts 4 \
    --part_method metis \
    --output ../data/partitions/photo_4_parts \
    --undirected \
    --balance_edges \
    --balance_train

if [ -d "../data/partitions/photo_4_parts" ]; then
    echo "✓ Partitioning complete"
    ls -lh ../data/partitions/photo_4_parts/ | head -10
else
    echo "❌ Partitioning failed"
    exit 1
fi
echo ""

# Train the model
echo "3️⃣  Starting distributed training..."
cd ../

LOGS_DIR="$PWD/logs/photo_training"
mkdir -p "$LOGS_DIR"

# Create IP config for single machine (format: IP PORT)
IP_CONFIG_FILE="$LOGS_DIR/ip_config.txt"
echo -n > "$IP_CONFIG_FILE"
echo "127.0.0.1 30050" >> "$IP_CONFIG_FILE"
echo "127.0.0.1 30051" >> "$IP_CONFIG_FILE"
echo "127.0.0.1 30052" >> "$IP_CONFIG_FILE"
echo "127.0.0.1 30053" >> "$IP_CONFIG_FILE"

# Use launch.py to start the DistDGL server and training client
echo "Running training with DistDGL launcher..."
conda run -n llm-dgl-cpu python3 launch.py \
    --workspace . \
    --num_trainers 1 \
    --num_samplers 0 \
    --num_servers 1 \
    --part_config data/partitions/photo_4_parts/photo.json \
    --ip_config "$IP_CONFIG_FILE" \
    --num_omp_threads 4 \
    --num_server_threads 1 \
    --workspace $PWD \
    "conda run -n llm-dgl-cpu python3 dist_gnn/main.py --graph_name photo --backend gloo --num_epochs 5 --batch_size 32 --lr 0.001 --num_gpus 0 --summary_filepath $LOGS_DIR/training_summary.txt --model sage --decision_model mlp --eviction false"

echo ""
echo "=========================================="
echo "✓ Pipeline complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - Partitions: data/partitions/photo_4_parts/"
echo "  - Logs: logs/photo_training/"
