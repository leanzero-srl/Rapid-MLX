#!/bin/zsh
# Launch one pipeline harness run across the MacBook (rank 0 host) and the
# workhorse.  Both machines must carry the same /tmp/flash-pipe layout:
#   /tmp/flash-pipe/repo    -> this fork checkout (with .venv)
#   /tmp/flash-pipe/model   -> the checkpoint directory
#   /tmp/flash-pipe/results -> a results directory
# Usage: flash_pipeline_launch.sh <jaccl|ring> <out-name> [harness pipe args...]
# mlx.launch exits 0 even when a rank dies: read rank<N>.log / rank<N>.json.
set -euo pipefail
backend=$1; name=$2; shift 2
hostfile=/tmp/flash-pipe/hostfile-$backend.json
if [[ $backend == jaccl ]]; then
  cat > $hostfile <<'JSON'
{"backend":"jaccl","hosts":[
 {"ssh":"127.0.0.1","ips":["192.168.0.1"],"rdma":[null,"rdma_en3"]},
 {"ssh":"workhorse","ips":["192.168.0.2"],"rdma":["rdma_en3",null]}]}
JSON
else
  cat > $hostfile <<'JSON'
{"backend":"ring","hosts":[
 {"ssh":"127.0.0.1","ips":["192.168.0.1"]},
 {"ssh":"workhorse","ips":["192.168.0.2"]}]}
JSON
fi
out=/tmp/flash-pipe/results/$name
mkdir -p $out
ssh workhorse "mkdir -p $out"
exec /tmp/flash-pipe/repo/.venv/bin/mlx.launch --hostfile $hostfile \
  --cwd /tmp/flash-pipe/repo -- \
  /tmp/flash-pipe/repo/.venv/bin/python scripts/flash_pipeline_harness.py pipe \
  --model /tmp/flash-pipe/model \
  --prompts /tmp/flash-pipe/results/FLASH-prompts.json \
  --out $out "$@"
