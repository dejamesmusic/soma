```
░░▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒░░       
░░▒▒▓▓                                                     ▓▓▒▒░░
░░▒▒▓▓                        soma                         ▓▓▒▒░░
░░▒▒▓▓                                                     ▓▓▒▒░░
░░▒▒▓▓                    usage guide                      ▓▓▒▒░░
░░▒▒▓▓                                                     ▓▓▒▒░░
░░▒▒▓▓      v8 · spectral online machine architecture      ▓▓▒▒░░
░░▒▒▓▓                                                     ▓▓▒▒░░
░░▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒░░

░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▓ bash ▓▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python soma_v8.py

░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▓ menu ▓▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░

on launch you'll see:

────────────────────────────────────────────────────────
    ░▒▓ soma ▓▒░
────────────────────────────────────────────────────────
  › mode (train/eval/chat):


░░░░░░░░░░░░░░░░░░░░░░░░░░▒▓ train ▓▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░

trains on any byte stream. point it at any file — text, code, 
binary. example 100M model:

  › mode (train/eval/chat): train
  › corpus: data/path
  › checkpoint (enter for new):
  › bands [46]: 46
  › range (base or window) [2500000000]: 1.6180
  › hidden (0=linear) [256]: 8192
  › lr [0.1]: 1.0
  › max_change [0.1]: 0.9
  › weight_decay [0.0001]: 0.0001
  › batch [50000]: 1000
  › decimation_band [0]: 0
  › direct readout (0/1) [0]: 1
  › epochs [1]: 1
  › start byte [0]: 0
  › report every [1000000]: 100000
  › save every (0=end) [0]: 1000000
  › save path [model.pt]: example.pt

parameters:

bands           number of temporal bands (K). more = finer 
                spectral resolution, wider feature vector. 
                default 46.

range           either the geometric base (if < 100) or the 
                maximum memory window in bytes (if ≥ 100). 
                base is computed from window via 
                base = window^(1/(K-1)). default 2.5B bytes.
                bases greater than 1.618 are discouraged.

hidden          hidden layer width (H). 0 for linear readout.
                parameter count ≈ H × 256K. default 256.

lr              learning rate. scales the raw gradient before
                clipping. use lr=1 to start.

max_change      maximum fractional change per weight per step.
                a weight of magnitude 0.01 can change by at 
                most max_change × 0.01 per step. use 1.

weight_decay    multiplicative decay applied after each update.
                W ← W × (1 - weight_decay). default 0.0001.

batch           number of samples per weight update. larger 
                batches give smoother gradients. default 50000.

decimation_band observation resolution. 0 = every byte. higher
                values skip bytes for throughput. 
                stride = base^d. default 0.

direct readout  if 1, adds Wd matrix bypassing the hidden layer.
                default 0.

epochs          passes over corpus. default 1.

start byte      resume position in the corpus. default 0.

report every    print progress every N bytes. default 1000000.

save every      checkpoint every N bytes. 0 = save at end only.

to resume from a checkpoint, enter its path when prompted. all
parameters from the checkpoint are loaded and can be overridden.

░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▓ eval ▓▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░

evaluates a checkpoint on a corpus. no weight updates. resets the
trace bank before evaluation.

  › mode (train/eval/chat): eval
  › checkpoint: model.pt
  › corpus: enwik8

reports: loss in nats, bits per byte (bpb), accuracy, throughput.

░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▓ chat ▓▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░

interactive generation from a checkpoint. your input is fed 
through the trace bank as context, then the model generates 
autoregressively.

  › mode (train/eval/chat): chat
  › checkpoint: model.pt
  › temperature [0.8]: 0.8
  › max length [200]: 200
  › online learning (y/n) [n]: y
  › lr [0.1]: 0.1
  › max_change [0.1]: 0.1

    temperature     controls randomness. lower = more 
                    deterministic. higher = more varied. 
                    default 0.8.

    max length      maximum bytes to generate per response.

    online learning if enabled, the model updates weights on 
                    your input bytes. the model learns from 
                    what you type.

commands during chat:

    quit / q        exit chat
    save            save current state to checkpoint

░░░░░░░░░░░░░░░░░░░░░░░░░░▒▓ files ▓▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░

    soma_v8.py           the implementation
    soma_v8_spec.md      complete algorithm specification
    requirements.txt     numpy, torch
    README.md            this file

░░░░░░░░░░░░░░░░░░░░░░░░░░▒▓ notes ▓▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░

requires python 3.8+ with numpy and torch. runs on cpu, cuda, or
mps. device is auto-detected.

trace bank runs in float64 on cpu regardless of device. the
forward path and weight updates run in float32 on the selected
device.

░░▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒░░
░░▒▒▓▓                                                     ▓▓▒▒░░
░░▒▒▓▓                                                     ▓▓▒▒░░
░░▒▒▓▓                 james blight © 2026                 ▓▓▒▒░░
░░▒▒▓▓                                                     ▓▓▒▒░░
░░▒▒▓▓                                                     ▓▓▒▒░░
░░▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒░░
```
