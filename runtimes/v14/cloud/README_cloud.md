# soma v13 cloud runs

the cloud harness builds corpora incrementally, trains the same v13 runtime as
the app, rotates recoverable checkpoints, and separates machine-readable
metrics from dreams.

## one run

```bash
python cloud/train_cloud.py \
  --data data.txt \
  --workdir run \
  --checkpoint soma_v13.pt \
  --bands 48 \
  --batch 1024 \
  --lr 0.0003 \
  --update-interval 1 \
  --resume
```

remove `--resume` to require a fresh checkpoint.

## staged curriculum

```bash
python cloud/run_cloud_curriculum.py \
  --plan cloud/curriculum_v1.json \
  --root . \
  --workdir run \
  --data data
```

the runner resumes only when corpus, checkpoint, and recovery cursor agree.
stage milestones are immutable copies. the active checkpoint also keeps one
rotated previous copy.

## observe

```bash
tail -f run/logs/metrics.jsonl
tail -f run/logs/dreams.txt
cat run/logs/status.json
cat run/logs/recovery.json
```

`metrics.jsonl` contains reports, saves, starts, completions, and failures.
`dreams.txt` contains generated samples only.

## checkpoint boundary

```text
species = soma_v13_trace_branch_mixer
```

earlier checkpoint species cannot resume in this runtime.
