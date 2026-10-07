#!/bin/bash

MODEL=${1:-gemma}

set -x
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
kill $PID
