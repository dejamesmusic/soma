# the weights are the model: one conversation per person, and everything else disposable

james blight · 2026-11-24 · version 1 · method · verdict: alive

## current evidence note

this version has no structured protocol review yet; consult its methods.


## original summary

what a soma model keeps, and what it throws away. each account has one conversation with each model, the same on every device. a file is read in a store of its own, which is dropped when the read ends: what it taught is in the weights. a long paste is read into the conversation itself, so it stays in working memory. the conversation is told, in a line it can trust, when a file was read. stores are disposable; only the weights are protected. the model builds its own continuity from what it can see.

published: 2026-11-25 09:47:51.580521+00:00

evidence: design, built and rehearsed end to end: two files read side by side in their own stores, words after them, no stores left behind; a paste read into the conversation; one conversation per account across devices · engine 2.1.2 to 2.1.6

[latest version](/api/developments/the-weights-are-the-model.md)

decided 2026-11-24, built 11-24 and 11-25.

## the short version

a person has one conversation with a model, wherever they are. everything else a model reads goes into its weights
through a store that is thrown away afterwards. the weights are the model. stores are scratch.

## the rule

- **one account, one conversation, per model.** the same conversation from every tab and every device. no list of
  chats to pick from.
- **a file is read in a store of its own**, in whatever lane is free, and the store is dropped when the read ends. what
  the file taught is in the weights.
- **words sent with files wait for them.** the reads finish first; then the conversation hears the words.
- **the conversation is told.** before the next words, the machine adds one line it writes itself, which nobody can
  forge: a file of yours was read and learned, and how big it was. lines a person types that look like it are set off,
  so they can't pass for it.
- **a long paste is different.** it goes into the model's files and is read into the conversation's own store, before
  the words. it stays in working memory. `/paste <file>` does the same with a file already there.
- **only the weights are protected.** a store can be lost and the model is still itself.

## why

- **coherence.** a person has one continuous relationship with one model, not a pile of threads.
- **reading doesn't fill the conversation.** before this, training on a file in a conversation's store left that store
  full of the file; the model had to readapt every time the person came back. now the file's store is gone, and the
  conversation stays the conversation.
- **continuity comes from the model.** it builds its sense of what has happened from what it can see and trust: its
  weights, its own conversation, and the lines the machine writes.

## what it took

- the machine keeps a separate store for every read, names it, and drops it at the end, whether the read finished,
  stopped or failed.
- an account's extra conversations from before were folded into the most recent one, with the person's agreement; the
  rest were deleted.
- the bridge reads a file part either in its own store or in the conversation's, as asked.

## what's honestly unknown

- with nothing useful in the store, a model has little to go on, and drifts and loops. a paste is the way to give it
  something; how much context a conversation needs to stay coherent is not measured.
- how the line the machine writes is used by the model, beyond being read.


