# soma-8k-v1

`soma-8k-v1.model` is the immutable language alphabet for soma v12.3.

- 8,192 sentencepiece bpe entries
- exact byte fallback
- identity normalization
- lexical pieces only; bpe does not cross whitespace boundaries
- no bos, eos, or padding token
- explicit soma protocol symbols

the model file and its sha256 hash are part of every v12.3 checkpoint's
compatibility contract. changing the vocabulary requires a new tokenizer id,
not an in-place replacement.

run `train_tokenizer.py` only to reproduce the published dictionary. production
models load the checked-in `.model`; they do not retrain it.

published sha256:

`9b435e2e58db184b35aae71af334f6ea097e2dc567f21e32ba5495c77d1cfcec`
