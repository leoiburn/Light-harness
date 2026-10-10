# Light-harness
A light cli for small local llms on linux

It's a minimal bash-only coding agent in C, in a single file. It's modeled on the text-based mode of
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent), and its prompts are adapted from that project.

How it works:

1. Send the history to an OpenAI-compatible `/chat/completions` endpoint.
2. Pull exactly one ```` ```mswea_bash_command ```` block out of the reply.
3. Run it with `/bin/sh -c` in a fresh process group, with stdin set to `/dev/null`. The command has a 30s timeout, and the whole group is killed on timeout. Output is capped at 10 KB.
4. Send `<returncode>` + `<output>` back as the next message, and repeat until the model prints
   `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.

## Build

```bash
sudo dnf install libcurl-devel cjson-devel   # Debian/Ubuntu: libcurl4-openssl-dev libcjson-dev
make
```

## Run

```bash
export MODEL=your-model-name
export OPENAI_API_KEY=...
# optional, for any OpenAI-compatible server, e.g. Ollama:
# export OPENAI_BASE_URL=http://localhost:11434/v1

./agent "count the lines of C in this directory"
./agent -y "..."   # skip the per-command confirmation (careful: runs on your machine)
```
