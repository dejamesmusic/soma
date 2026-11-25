# a language for learning machines

james blight · 2026-11-25 · version 2 · method · verdict: alive

## current evidence note

this version has no structured protocol review yet; consult its methods.


## original summary

logOS is a terminal, and the terminal is a language: every page, link and address resolves to one command, and every command starts with /. it opens blank. each account has one terminal stream, kept on logOS and followable live from any device or program. programs get their own keys, each with only the powers its owner gives it. the same language runs over http (/api/run), for agents through an mcp server, and from a command line (logos), so anything can be built on it.

published: 2026-11-25 20:48:38.930117+00:00

evidence: built and deployed nov 25–26: language v1, the stream and its filters, personal access tokens and their scopes, /api/run, the mcp server, the logos command line; tested (terminal and backend suites, an access sweep over every route); a conversation over /api/run not yet tried against a running model

[latest version](/api/developments/a-language-for-learning-machines.md)

built 2026-11-25.

## the short version

the value is the system, not the interface. so the interface is a language: small, consistent, and the same for a
person typing, a program calling, or an agent acting.

## the rules

- **a command starts with `/`.** anything else is written to the stream: said to the model you're talking to, or, with
  no model open, just written.
- **one line, one command.** a command missing what it needs asks for it, one question at a time: `/move` asks which
  model, `/move soma-1b` asks where.
- **every command shows as typed**, however it arrived: typed, clicked, from an address, from a key.
- **it opens blank.** nothing is explained first. `/help` explains.
- **names, not labels.** speakers are the model's name and yours. usernames are chosen, never taken from an email.

## the stream

- one per account. every command and what it printed, kept on logOS (the newest 2,000 blocks).
- a new terminal starts blank; `/history` fills it in, from every device.
- every change to a block is numbered, so any client can follow the stream live: what changed since the last number.
- it doubles as the audit log: anything a program's key changes is written into it, under the key's name.

## keys for programs

- `/token new` makes a key with only the powers given: **read**, **talk**, **manage**, **spend** (with a monthly
  cap, counted from the machines it starts).
- no key may ever make keys, delete a model, close an account, change a username or touch a runpod key.
- the secret is shown once and kept out of every history. logOS stores a hash.

## beyond the browser (nov 26)

- **the language over http.** `POST /api/run {"command": "/models"}`: one call, one line, back comes what the terminal
  would draw, as segments and plain text, plus the json the command fetched, any question it asks, and a session that
  remembers the open model and the last list. the browser's own code runs on the server, one short process per call,
  so the two can't drift. it reaches logOS with the caller's own credentials: a key can do there exactly what it could
  do anywhere else.
- **agents, through mcp**, at `logossoma.com/mcp`: the whole language as one tool, and typed tools for the common work
  (talk, read a file, paste a document, stop, follow the stream). one mcp session is one terminal session.
- **a command line**, `logos`: the same language in any terminal, a line at a time for scripts, or following the
  stream live.
- **the stream, filtered**: only replies, only reads, only one model.

still to come: one-click sign-in for agents, a plain chat page, and a versioned api.


