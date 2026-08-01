# soma v12.3.4 cloud runs

the cloud harness uses the same tokenizer, checkpoint and runtime as the mac
app. keep the complete script bundle together; `tokenizer/soma-8k-v1.model` is
required beside the runtime.

example fresh run:

```sh
python cloud/train_cloud.py \
  --data /workspace/data/enwik9 \
  --workdir /workspace/soma-run \
  --checkpoint soma_70m.pt \
  --bands 16 --depth 4 --batch 512 \
  --lr "auto 0.002" --decimation 1 --epochs 1 \
  --dream-every 64 --dream-tokens 128 --device cuda
```

resume the same checkpoint:

```sh
python cloud/train_cloud.py \
  --data /workspace/data/enwik9 \
  --workdir /workspace/soma-run \
  --checkpoint soma_70m.pt --resume --device cuda
```

outputs:

```text
checkpoints/<name>.pt
checkpoints/<name>.pt.meta.json
logs/metrics.jsonl
logs/status.json
logs/dreams.txt
logs/environment.json
```

`status.json` is safe to poll while training. checkpoints are saved atomically.
interrupting the trainer saves once before exit.
