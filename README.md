# Q-SOTA

Automated literature review from arXiv, using large language models as a judge to determine
relevancy and quality of papers.

## Usage

To use this, you have to clone the repo, (recommended) set up a virtual environment, and run the
python script:

```bash
git clone https://github.com/owahltinez/qsota.git && cd qsota
python -m venv .venv
source .venv/bin/activate
python qsota.py <command>
```

## Commands

For the exhaustive list of parameters that each command supports, run:

```bash
python qsota.py <command> help
```

Commands that make use of LLMs evaluate prompts via the
[llm package](https://pypi.org/project/llm/). If you want to use models that require further setup,
then you'll need to do that first. For example, to use Gemini models:

```bash
python -m pip install llm
llm install llm-gemini
llm keys set gemini
```

All commands that use an LLM accept `--mode_name` and `--model_key` parameters, as well as `--qps`
(queries per second) for rate limiting.

### `search`

Searches arXiv using the [arXiv API](https://info.arxiv.org/help/api/index.html). No API token
or account are required. The query parameter corresponds to the handling of arXiv search
functionality as-is. Example:

```bash
python qsota.py search --query='all:wfh' > search_results.jsonl
```

The output are JSON lines, so you might want to pipe the output to a file `> search_results.jsonl`
or to one of the other commands which accept JSON lines from `stdin` as input.

### `relevant`

Evaluates a relevancy score [0-1] for the given list of papers, and provides a rationale for the
score. This command uses **only** the title, abstract and authors of the paper to determine whether
it is relevant or not. It also requires a `query` parameter that describes, in plain words, what
is the research that the relevancy score is being evaluated for. Example:

```bash
CONTEXT=$(cat << EOF
Investigating the influence of pet interruptions on productivity and stress levels in remote work.
EOF
)
cat search_results.jsonl | python qsota.py relevance --query "$CONTEXT" > relevance.jsonl
```

Once again, the output are JSON lines. So you might want to pipe the output to a file
`> relevancy.jsonl` or to one of the other commands which accept JSON lines from `stdin` as input.

### `quality`

Provides a quality score [0-1] for the given list of papers, and provides a rationale for the score.
This command will also download the full text for the paper, so it is expected to be significantly
slower. Example:

```bash
cat relevance.jsonl | python qsota.py quality > quality.jsonl
```

Once again, the output are JSON lines. So you might want to pipe the output to a file
`> quality.jsonl` or to one of the other commands which accept JSON lines from `stdin` as input.

## Chaining

The provided commands are designed to be chained together, so you can for example run:

```bash
CONTEXT=$(cat << EOF
Investigating the influence of pet interruptions on productivity and stress levels in remote work.
EOF
)

python qsota.py search --query='all:wfh' \
    | python qsota.py relevance --query "$CONTEXT" \
    | python qsota.py quality \
    > shortlist.jsonl
```
