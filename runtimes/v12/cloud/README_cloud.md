# soma v12 cloud runner

minimal vast workflow:

```bash
cd /workspace/soma_v12
python cloud/train_cloud.py \
  --data /workspace/soma_v12/data/enwik9 \
  --checkpoint soma_cloud.pt \
  --bands 50 \
  --hidden 1024 \
  --layers 3 \
  --batch 512 \
  --auto-mode io2 \
  --lr auto \
  --max-change auto \
  --decimation 1.0 \
  --save-minutes 30 \
  --dream-every 50
```

explicit controller strings are also accepted:

```bash
  --lr "io2 1.0" \
  --max-change "io2 1.0" \
```

files written:

- `checkpoints/soma_cloud.pt`
- `checkpoints/soma_cloud.bak1.pt`
- `checkpoints/soma_cloud.bak2.pt`
- `logs/metrics.jsonl`
- `logs/status.json`
- `logs/dreams.txt`
- `logs/environment.json`

monitoring:

```bash
tail -f logs/metrics.jsonl
tail -f logs/dreams.txt
cat logs/status.json
```

safe restart:

```bash
cd /workspace/soma_v12
python cloud/train_cloud.py --data /workspace/soma_v12/data/enwik9 --checkpoint soma_cloud.pt
```

it resumes automatically if the checkpoint exists.
