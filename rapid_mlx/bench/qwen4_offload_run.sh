#!/bin/zsh
# One offload config: start our own server, run the bench client, stop OUR server (by pid).
# usage: qwen4_offload_run.sh <label> <outdir> <prompts> [extra serve args...]
# env: PY (python), MODEL (dir), PORT (8197), MTP=1|0, REPEAT (1)
set -u
label=$1; outdir=$2; prompts=$3; shift 3
WT=${WT:-$HOME/Projects/Rapid-MLX-wt-offload}
PY=${PY:-$HOME/.cache/uv/archive-v0/7bolWjpGXF64QzWe/bin/python}
MODEL=${MODEL:-$HOME/models-local/Qwen3.8-Flash-Next-4bit}
PORT=${PORT:-8197}
mkdir -p $outdir
spec=()
if [[ ${MTP:-1} == 1 ]]; then
  spec=(--speculative-config "{\"method\":\"mtp\",\"model\":\"$MODEL\",\"num_speculative_tokens\":${MTP_K:-2}}")
fi
t0=$(date +%s)
PYTHONPATH=$WT $PY -m rapid_mlx.cli serve $MODEL --port $PORT --text-only \
  --max-concurrent-requests 8 --enable-prefix-cache --hybrid-cache-entries 24 \
  $spec "$@" > $outdir/$label.server.log 2>&1 &
spid=$!
trap 'kill $spid 2>/dev/null; wait $spid 2>/dev/null' EXIT
for i in {1..900}; do
  curl -sf http://127.0.0.1:$PORT/health/ready >/dev/null 2>&1 && break
  kill -0 $spid 2>/dev/null || { echo "$label SERVER DIED"; tail -5 $outdir/$label.server.log; exit 1; }
  sleep 2
done
t1=$(date +%s)
echo "$label load_s=$((t1-t0)) $(curl -s http://127.0.0.1:$PORT/v1/status | $PY -c 'import json,sys;s=json.load(sys.stdin);print("idle_metal",s.get("metal"),"offload",s.get("qwen4_offload"))')"
PYTHONPATH=$WT $PY -m rapid_mlx.bench.qwen4_offload_bench --port $PORT --label $label \
  --prompts $prompts --repeat ${REPEAT:-1} --out $outdir/$label.json
rss=$(ps -o rss= -p $spid); echo "$label rss_gb=$((rss/1048576))"
