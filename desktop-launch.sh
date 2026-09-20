#!/bin/bash

# swap out build-arg for CPU if doing that.
docker build -t localhost:rudder-llm-agent:latest-gpu -f docker/Dockerfile --build-arg=TYPE=gpu .

# inside the running container.
ollama serve &
PID=$!
ollama pull gemma # from the decision model argument
kill $PID

python3 launch.py \
	--workspace $PWD \
	--num_trainers 1 \
	--num_samplers 1 \
	--num_servers 1 \
	--part_config $PWD/../partition-output1/photo.json \
	--ip_config $PWD/ip_config.txt \
	--num_omp_threads 1 --num_server_threads 1 \
	'source /etc/profile && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && /opt/conda/condabin/conda run -n llm-dgl-cuda121 python3 dist_gnn/main.py --graph_name photo --backend gloo --num_epochs 5 --batch_size 8 --lr 0.001 --num_gpus 1 --summary_filepath training_summary.txt --model sage --decision_model gemma --ip_config ip_config.txt --part_config ../partition-output1/photo.json'
