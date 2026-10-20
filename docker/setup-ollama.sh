#!/bin/bash

MODEL=${1:-gemma}

eval "$(conda shell.bash hook)"
set -x
conda activate llm-dgl-cuda121
# inside the running container.
ollama serve &
PID=$!

# Wait for ollama to startup
echo "Waiting for Ollama to start..."
while ! curl -s http://localhost:11434/api/tags > /dev/null; do
    sleep 1
done
echo "Ollama is ready!"

ollama pull "$MODEL" # from the decision model argument
if ! kill $PID; then
    echo "Failed to kill Ollama process."
    exit 0
fi
echo "Ollama process terminated."
exit 0