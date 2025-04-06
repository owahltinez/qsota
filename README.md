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
[langfun package](https://pypi.org/project/langfun/). Models require either passing an API key as
an argument, or the API key can be in one of the known environment variables i.e., `OPENAI_API_KEY`
or `GOOGLE_API_KEY`.

All commands that use an LLM accept `--mode-name` and `--model-key` parameters.

### `search`

Searches a paper database. The currently supported databases are:

* [arXiv][arxiv] (`arxiv`): [arXiv API][arxiv-api]. No API token or account required.
* [Astrophysics Data System][ads] (`ads`): [ADS API][ads-api]. Token required for authentication.

Example:

```bash
python qsota.py search --query='wfh' --database='arxiv' > search_results.jsonl
```

The output are JSON lines, so you might want to pipe the output to a file `> search_results.jsonl`
or to one of the other commands which accept JSON lines from `stdin` as input.

You can use the optional flag `--has-doi` to filter results which have an external
[DOI](https://doi.org) associated with them. This means that the result is likely to have been
published elsewhere.

[arxiv]: https://arxiv.org/
[arxiv-api]: https://info.arxiv.org/help/api/index.html
[ads]: https://ui.adsabs.harvard.edu/
[ads-api]: https://github.com/adsabs/adsabs-dev-api

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
cat search_results.jsonl \
    | python qsota.py relevance --query "$CONTEXT" --threshold=0.1 \
    > relevance.jsonl
```

Once again, the output are JSON lines. So you might want to pipe the output to a file
`> relevancy.jsonl` or to one of the other commands which accept JSON lines from `stdin` as input.

### `quality`

Provides a quality score [0-1] for the given list of papers, and provides a rationale for the score.
This command will also download the full text for the paper, so it is expected to be significantly
slower. Example:

```bash
cat relevance.jsonl \
    | python qsota.py quality --threshold=0.1 \
    > quality.jsonl
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

python qsota.py search --query='wfh' --database='arxiv' \
    | python qsota.py relevance --query="$CONTEXT" --threshold=0.1 \
    | python qsota.py quality --threshold=0.1 \
    > shortlist.jsonl
```

## Using `pipx`

You can also install this package using [`pipx`](https://github.com/pypa/pipx):

```bash
pipx install 'git+https://github.com/owahltinez/qsota.git'
```

Then you can use this script as a command line tool under `qsota`:

```bash
CONTEXT=$(cat << EOF
Investigating the influence of pet interruptions on productivity and stress levels in remote work.
EOF
)

qsota search --query='wfh' --database='arxiv' \
    | qsota relevance --query="$CONTEXT" --threshold=0.1 \
    | qsota quality --threshold=0.1 \
    > shortlist.jsonl
```

