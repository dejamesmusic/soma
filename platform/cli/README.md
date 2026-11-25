# logos

the logOS terminal in your own. `npm i -g logossoma`, then `logos`. the language runs on logOS (`POST /api/run`), the same code as the browser's.

```
logos                      the terminal
logos run "/models"        one line: --json, --confirm, --wait <s>, --session <id>
logos stream               follow your stream: every terminal, program and agent
logos login                sign in through your browser (or --token lgp_... for a key from /token new)
logos logout
```

`/paste <file>` sends a local file to the model you're talking to. `LOGOS_URL` picks the site; `LOGOS_TOKEN` overrides
the saved sign-in (`~/.config/logos/config.json`, readable only by you).
