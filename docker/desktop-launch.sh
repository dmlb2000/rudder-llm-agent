#!/bin/bash

set -x
# swap out build-arg for CPU if doing that.
# docker build -t localhost:rudder-llm-agent:latest-gpu -f docker/Dockerfile --build-arg=TYPE=gpu .

cat $MPI_HOSTSFILE | awk '{ print $1 }' > $PWD/ip_config.txt
DS_NAME="${1}"
PART_CONFIG="${2}"
DECISION_MODEL="${3:-gemma}"
GPUS_PER_NODE="${4:-1}"
NUM_SAMPLERS="${5:-2}"
NUM_SERVERS="${6:-2}"

# These are passed from the IRI-API?
HOST_PROJECT_MOUNT_PATH=$HOST_PROJECT_MOUNT_PATH
CONTAINER_PROJECT_MOUNT_PATH=$CONTAINER_PROJECT_MOUNT_PATH
# Just your home directory resolved
HOME_PATH=$HOME

# get total number of cores on the node (multiply by number of sockets)
CORES_PER_SOCKET=$(lscpu | grep "Core(s) per socket" | awk '{print $4}')
SOCKETS=$(lscpu | grep "Socket(s)" | awk '{print $2}')
TOTAL_CORES=$(($CORES_PER_SOCKET * $SOCKETS))
TRAINERS=$GPUS_PER_NODE

CORES_PER_TRAINER=$(($TOTAL_CORES / $TRAINERS))
OMP_THREADS=$(($CORES_PER_TRAINER)) # OMP threads = number of cores per trainer
# numba threads = OMP threads - 1
NUMBA_THREADS=$(($OMP_THREADS)) # all cores to numba threads as gpu is used for training

echo "Total Cores: $TOTAL_CORES"
echo "Cores per Trainer: $CORES_PER_TRAINER"
echo "OMP Threads: $OMP_THREADS"
echo "Numba Threads: $NUMBA_THREADS"

echo "JOBID: $JOBID"

python3 launch.py \
	--workspace $PWD \
	--num_trainers $GPUS_PER_NODE \
  --num_samplers $NUM_SAMPLERS \
  --num_servers 1 \
	--part_config $PART_CONFIG \
	--ip_config $PWD/ip_config.txt \
	--num_omp_threads $OMP_THREADS \
	'singularity exec 
	  --mount type=bind,source=/tmp,target=/tmp 
		--mount type=bind,source='$HOME_PATH',target='$HOME_PATH' 
		--mount type=bind,source='$HOST_PROJECT_MOUNT_PATH',target='$CONTAINER_PROJECT_MOUNT_PATH'
		conda run -n llm-dgl-cuda121 python3 dist_gnn/main.py 
		  --graph_name '$DS_NAME' 
			--backend gloo 
			--num_epochs 5 
			--batch_size 8 
			--lr 0.001 
			--num_gpus '$GPUS_PER_NODE' 
			--num_numba_threads '$NUMBA_THREADS' 
			--summary_filepath training_summary.txt 
			--model sage 
			--decision_model '$DECISION_MODEL' 
			--ip_config ip_config.txt 
			--part_config '$PART_CONFIG
