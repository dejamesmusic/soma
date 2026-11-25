# the logOS language

logOS is a terminal, and the terminal is a language: every page, link and address resolves to one command, and every
command starts with `/`. the same language serves a person typing, a program calling and an agent acting. the full
reference: [`LANGUAGE.md`](LANGUAGE.md); every command: `/help`.

| | |
|---|---|
| [`language/`](language) | the language itself: the code the browser terminal runs (`shell.js` holds the rules for running a line) |
| [`server/run.js`](server/run.js) | the same code run headless by logOS for `/api/run`, one short process per call |
| [`cli/`](cli) | `logos`, the terminal in your own: `npm i -g logossoma` |

## for programs

```
POST https://logossoma.com/api/run   {"command": "/public models"}
```

back comes what the terminal would draw, as segments and as plain text, the json the command fetched, any question it
asks, and a `session` that remembers the open model and the last list. answer a question with
`{"answer": "...", "session": "..."}`, or give the whole command. `"confirm": true` answers the terminal's own [y/n]
confirmations; `"wait": 60` stays for a model's reply to words.

signed out, a program sees what anyone sees. signed in, it uses a personal access token made with `/token new`,
carrying only the scopes it's given: `read`, `talk`, `manage`, and `spend` with a monthly cap. no token can make tokens,
delete a model, close an account, change a username or touch a runpod key. everything a token changes is written into
its owner's stream.

## the live stream

```
GET https://logossoma.com/api/me/terminal?after=<rev>&wait=25&kind=reply&model=<name>
```

every block any terminal, program or agent drew for an account, since a number. kinds: command, you, reply, read,
skill, note, audit.

## for agents

an mcp server at `https://logossoma.com/mcp` (streamable http), signed in with a personal access token. tools: `run`
(the whole language), `stream`, `list_models`, `talk`, `read_file`, `paste`, `stop`. one mcp session is one terminal
session.

the language code here is MIT, like everything in this repository. the logOS service behind it is not open source.
