# soma script bundle

This is the terminal version of soma. It is for people who want to run soma in their own Python environment, on a rented GPU, on Linux, or anywhere the macOS app is not the right fit.

If you just want the easiest macOS experience, use the soma DMG instead.

## contents

```text
soma/
  soma                  launcher script
  soma_v10.py           interactive train / eval / chat runtime
  soma_loop.py          perpetual training helper
  streams_registry.py   discovers stream scripts
  _impl_torch.py        torch backend
  _impl_metal.py        apple metal backend
  _impl_triton.py       triton backend
  requirements.txt      Python dependencies
  data/                 put corpus text files here
  data/streams/         default output location for stream corpora
  checkpoints/          put .pt checkpoint files here
  streams/              stream scripts, including wikipedia.py
  tools/                corpus preparation helpers
  docs/                 technical spec
```

No checkpoints are included. Download checkpoints from logOS.

## quick start

From a terminal:

```sh
unzip soma-script-bundle.zip
cd soma
./soma
```

On first run the launcher creates a local `venv` folder and installs `numpy<2` and `torch`.

If your machine does not preserve the executable bit after unzipping:

```sh
chmod +x soma
./soma
```

## manual environment

On a rented GPU or server, you may prefer to manage Python yourself:

```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python soma_v10.py
```

If your GPU host needs a specific CUDA build of PyTorch, install that build first using the command recommended by PyTorch for the machine, then run:

```sh
pip install "numpy<2"
python soma_v10.py
```

## paths

soma has simple path conventions:

- Bare corpus filenames are looked up inside `data/`.
- Bare checkpoint filenames are looked up inside `checkpoints/`.
- Explicit paths are used exactly as typed.
- Absolute paths are fine.

Examples:

```text
my_corpus.txt              -> data/my_corpus.txt
priant.pt                  -> checkpoints/priant.pt
data/streams/wikipedia.txt -> data/streams/wikipedia.txt
/workspace/checkpoints/a.pt -> /workspace/checkpoints/a.pt
```

That means rented GPU workflows are fine: upload your checkpoint and corpus wherever you want, then enter the absolute path.

## interactive soma

Run:

```sh
./soma
```

Then choose `train`, `eval`, or `chat` when prompted.

Typical starter layout:

```text
soma/
  data/my_corpus.txt
  checkpoints/my_model.pt
```

For a new model, leave the checkpoint/load field empty when training, then save to a checkpoint name such as `my_model.pt`.

## streams

List installed streams:

```sh
./soma streams
```

Run the built-in Wikipedia stream:

```sh
./soma stream wikipedia
```

With no output path, it writes to:

```text
data/streams/wikipedia.txt
```

You can also choose your own output:

```sh
./soma stream wikipedia data/wiki.txt
```

To train continuously against a rolling stream corpus:

```sh
./soma loop data/streams/wikipedia.txt checkpoints/my_model.pt
```

## adding stream accessories

Any Python file dropped into `streams/` that declares these constants will appear in `./soma streams`:

```python
STREAM_NAME = "my_stream"
STREAM_DESCRIPTION = "what this stream produces"
```

A stream should write plain UTF-8 text to a corpus file. If no corpus path is provided, the usual convention is `data/streams/<stream_name>.txt`.

## corpus tools

Two small helpers are included in `tools/`:

```sh
python tools/concat_txt_corpus.py
python tools/build_fineweb_edu_txt.py
```

Use these to prepare plain-text corpora before training.

## notes

- Checkpoints are portable `.pt` files.
- Keep large corpora in `data/` or use absolute paths on external/server storage.
- Keep generated checkpoints in `checkpoints/` or use absolute paths.
- The local `venv/` created by the launcher is intentionally not included in this zip.
