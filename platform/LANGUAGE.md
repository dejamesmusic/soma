# The logOS language, as it stands (Nov 25, 2026)

The terminal is a small programming language over logOS, the soma engine, RunPod and the machines. Every page,
link, button and URL resolves to one typed command, and the command is printed as typed. This file lays the language
out and audits it. It is the groundwork for serving the language from logOS itself (`POST /api/run`), so other
interfaces can be built on it; nothing here is frozen until James says so.

## 1. How it reads (v1, Nov 25)

| rule | what it means |
|---|---|
| a command starts with `/` | everywhere. A line without `/` is the stream: said to the model you're talking to, or, outside a conversation, written to your stream (one day, a model's input too). Clicks type the slash. |
| one stream per account | everything the terminal draws is kept on logOS (newest 2000 blocks). A new terminal starts blank; `/history` fills it in up to now, from every device; `/history clear` deletes it. |
| a blank start | logOS opens on nothing but a prompt. `/home` is the title and its doors; only `/help` explains. An address still opens its page; `/set start` still runs. |
| one line, one command | verb first, then its object: `/share tiny with @bob`. A command missing what it needs asks for it, one question at a time: `/move` asks which model, `/move tiny` asks where. |
| names and numbers | a model is named by its name, its id, or `n` from the last list (`/open 2`). Reads are numbered while they run (`/stop 2`). |
| questions | a question's prompt ends in `>`; the next line answers it, without a slash. Esc cancels. |
| echo | every command shows as typed, with its prompt: typed, clicked, from an address, or a key (Esc = `/exit`). Ctrl-L (`/clear`) doesn't. Secret answers show as `[hidden]`. |
| status | the bottom lines are live: one row per running read of yours, then the model's line. |
| names | speakers are names: the model's and yours. Usernames are picked, never taken from an email. |

## 2. The words

**Nouns.** model (weights, the thing that learns), library (your models), publication (a frozen public snapshot),
place (where a model is stored: logOS, a computer, a volume), machine (a computer or GPU that runs models), volume
(cloud storage in a data center), chat (your one conversation with a model; one per account per model), store (a
chat's working memory; disposable), file (a model's data), read (a file going through the model), lane (one thing a
machine runs at once), seat (a person using a model), guest (a seat on someone else's model), budget (what a model's
machines may spend).

**Verbs.** open, search, star, fork, add, rename, publish, move, load, unload, remove, delete, share, unshare,
talk, read, pause, resume, skip, stop, save, backup, score, learn, link, unlink, sign in, sign out, set.

## 3. The commands

The command list is `frontend/src/terminal/help.json`, served as `/api/help.md` and printed by `/help`. One source,
so the terminal, its help and its machine-readable twin can't drift apart.

## 4. The audit

**Decided Nov 25 (James), built in v1:** `/exit` only leaves; `/history` is the terminal stream (`/history <model>`
and `/cloud history` keep theirs); `/delete model` deletes every copy wherever it is (unloaded first, with warnings);
`remove` is gone; one numbered `/stop` (`/stop` the answer, `/stop n` a read; `/resume n`, `/skip n`); one `move`
verb (no `cloud move`, `cloud home`, `cloud stop`); every command needs `/`, so bare words are never commands;
*machine* for anything that runs a model; speakers are names; usernames picked at first sign-in.

**Still open:** a pass over every output to strip explanation, leaving only `/help` to explain (the home page and
settings are done); `cloud stop <machine>` survives inside the code for `/unload` of a cloud model, and machines that
aren't chats (copies, moves) have no stop of their own in the language; `/rename` and `/share` don't yet ask for what
they're missing; `/pause n` still works as an old spelling of `/stop n`.

The findings below are the audit as written before those decisions.

Each finding is marked **fixed** (done in this round), or **decide** (it changes the language, so it needs James's
call). Proposals are mine.

### Contradictions

1. **fixed:** the help still described several chats per model (`/chats`, `/new chat`, `/chat <n>`, `/discard`,
   "your chats, each its own store") after one chat per account (Nov 24).
2. **fixed:** `/exit` was described as "save it and stop it", but in lanes, or on a shared model, it leaves and the
   model keeps running. The help now says so. **decide:** make `/exit` always just leave. With one chat across
   every tab and device, one tab closing shouldn't stop the model for the others; `/unload` is the one way to stop.
3. **decide:** `load <model>` is listed as an alias of `talk to`, but it's its own command (it asks where, then loads).
   Proposal: `load` only loads; `talk to` loads if needed and opens the chat.
4. **decide:** `chats <model>` says "your chats with X" (plural) for what is one chat. Proposal: `chat <model>`
   shows the one chat; `chats` lists them all, one per model.

### One word, several meanings

5. **history** means five things: what you typed (`history`), a model's events (`history <model>`), this chat
   (`/history`), a chat by number (`chat history <n>`) and your cloud machines (`cloud history`). Proposal: keep
   `history <model>` and `cloud history`; `/history` becomes `/chat`, `chat history <n>` becomes `chat <n>`, and the
   typed list becomes `typed` (or the up arrow, which already works).
6. **home** means the start page (`home`, `set home`) and logOS as a model's place (`cloud home <model>`).
   Proposal: `cloud home <model>` becomes `move <model> to logOS`, which already exists. Drop the duplicate.
7. **stop**: `/stop` stops the current answer (and pauses a read), `cloud stop` ends a machine, and `/exit` used
   to "save and stop". Proposal: `/stop` is only "stop the answer"; reads use `/pause`; machines use
   `cloud stop`.
8. **remove**: `remove <model> from <place>` removes one copy, but `remove <model>` deletes the model (it asks
   first). An easy slip. Proposal: `remove` always needs `from <place>`; deleting is only `delete`.
9. **cat / less / more** open a development, while `cd` / `open` open a model. Proposal: drop the unix
   aliases for developments, or make `cat <x>` open whatever `<x>` names.
10. **profile** is both you (`profile`) and someone else (`profile @name`). That's fine, since the `@` decides;
    keep it.
11. **move**: `move <model> to <place>` and `cloud move <model> to <dc>` overlap. Proposal: one verb,
    `move <model> to <place>`, where a place can be a data center.

### Dead ends and easy mistakes

12. **decide:** in a conversation, typing a command without its slash (`help`, `exit`, `clear`) is speech, sent to
    the model. Proposal: those three bare words, alone on a line, ask "say it to the model, or run the command?"
    (Enter says it.)
13. **fixed:** pausing or skipping one of several reads meant pausing all of them, and the controls were back in the
    scrollback. Now `/pause 2`, `/resume 2` and `/skip 2` act on one read; bare `/pause` acts on all of yours; each
    read has a live status row; `/lanes` shows everything.
14. **fixed:** commands typed in a conversation (`/pause`, `/files`), the opening `home`, and Esc's `/exit` ran
    without showing as typed. Every command now echoes.
15. Outputs that end in "kept." or "done." with nothing to click: each answer should end with the next useful
    command. That needs a pass over all of the outputs; it's next.

### Wording

16. **machine / computer / mac / device** all appear. Proposal: *machine* means anything that runs a model;
    *computer* means yours (the soma app); *mac* goes, since Windows exists.
17. **soma** is the app, the engine, and the model's speaker label (`soma>`). Proposal: the speaker is the
    model's name.
18. **chat** is a verb (`chat to`) and a noun (your chat). It's fine in English; keep `talk to` as the verb in
    suggestions.
19. **share … to teach** and **as teacher** both work. Keep both, and document `to teach`.

### Identity

20. **decide:** a new account's username is the local part of its Google email (`jane.doe42`), which shows on
    every public page. That leaks part of the email. Proposal:
    - a new account gets a neutral name (`user-7f3a2c`), and the first sign-in asks for a username: 3–30
      characters, `a-z 0-9 . _ -`, unique, with a few reserved words (admin, logos, soma, api, …);
    - `set username <name>` changes it at any time, and `settings` shows it;
    - an account whose username is still its email's local part is asked once, at its next sign-in;
    - an old name is never given to anyone else, so links and shares to it can't land on a stranger; links to an
      old name forward to the new one.
