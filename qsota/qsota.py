import json
import re
import os
from typing import Iterable, TypedDict, TypeVar
import xml.etree.ElementTree as ET

import click
import httpx
import langfun as lf
import magic


T = TypeVar("T")
S = TypeVar("S")

_DEFAULT_MODEL_NAME = "google_genai://gemini-2.0-flash"
_DEFAULT_SEARCH_LIMIT = 100
_DEFAULT_THRESHOLD = 0.75


# Path to the folder containing this script.
_CURR_DIR = os.path.dirname(os.path.abspath(__file__))

_RELEVANCE_SCORE_PROMPT_PATH = f"{_CURR_DIR}/relevance_score_prompt.txt"
_QUALITY_SCORE_PROMPT_PATH = f"{_CURR_DIR}/quality_score_prompt.txt"


class Submission(TypedDict):
  id: str
  content_uri: list[str]
  authors: list[str]
  title: str
  categories: list[str]
  abstract: str
  date: str
  publication: str | None
  doi: list[str]


class RelevanceResponse(TypedDict):
  relevance_score: float
  relevance_rationale: str


class QualityResponse(TypedDict):
  quality_score: float
  quality_rationale: str


def _normalize_text(text: str) -> str:
  return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()


@click.group()
def cli() -> None:
  pass


def _iter_stdin_json() -> Iterable[dict]:
  for line in click.get_text_stream("stdin"):
    if not line.strip():
      continue
    try:
      yield json.loads(line)
    except json.JSONDecodeError as exc:
      raise RuntimeError(f"Error decoding JSON: {exc}. Line: {line}") from exc


def _parse_json_response(text: str) -> dict:
  text = text.strip()
  text = text[text.find("{") : text.rfind("}") + 1]
  try:
    return json.loads(text)
  except json.JSONDecodeError as exc:
    raise RuntimeError(f"Error decoding JSON: {exc}. Text: {text}") from exc


def _iter_arxiv_api(
    query: str | None = None,
    limit: int | None = None,
) -> Iterable[Submission]:
  """
  Searches arXiv using the provided query and yields results as they are processed.
  Uses httpx for making the HTTP request.

  Args:
    query: The search query string.
    limit: (Optional) The maximum number of results to retrieve.

  Yields:
    Iterable of Submission objects, where each object represents a paper.
  """

  base_url = "http://export.arxiv.org/api/query"
  limit = _DEFAULT_SEARCH_LIMIT if limit is None else limit
  params = dict(search_query=query, start=0, max_results=limit)
  response = httpx.get(base_url, params=params, timeout=300).raise_for_status()

  root = ET.fromstring(response.content)
  ns = "{http://www.w3.org/2005/Atom}"

  for entry in root.findall(f"{ns}entry"):
    id = entry.find(f"{ns}id").text.split("/")[-1]
    author_list = entry.findall(f"{ns}author")
    category_list = entry.findall(f"{ns}category")
    doi = entry.find(f"{ns}link[@title='doi']")
    doi = [doi.attrib["href"].split("doi.org/")[-1]] if doi is not None else []
    yield Submission(
        id=id,
        # content_uri=[f"https://ar5iv.org/pdf/{id}"],
        content_uri=[f"https://arxiv.org/pdf/{id}"],
        authors=[x.find(f"{ns}name").text for x in author_list],
        title=_normalize_text(entry.find(f"{ns}title").text),
        categories=[x.attrib["term"] for x in category_list],
        abstract=_normalize_text(entry.find(f"{ns}summary").text),
        date=entry.find(f"{ns}updated").text[:10],
        doi=doi,
    )


def _iter_ads_api(
    query: str | None = None,
    limit: int | None = None,
    token: str | None = None,
) -> Iterable[Submission]:
  base_url = "https://api.adsabs.harvard.edu/v1/search/query?"
  limit = _DEFAULT_SEARCH_LIMIT if limit is None else limit
  params = dict(
      q=query,
      rows=limit,
      fl="bibcode,title,author,keyword,abstract,date,pub,doi",
  )
  response = httpx.get(
      base_url,
      params=params,
      headers={"Authorization": f"Bearer {token}"},
      timeout=300,
  ).raise_for_status()

  response_json = response.json()
  if response_json["response"]["numFound"] == 0:
    raise RuntimeError("No results found")
  for doc in response_json["response"]["docs"]:
    authors = doc.get("author", [])
    if isinstance(authors, str):
      authors = [authors]
    yield Submission(
        id=doc["bibcode"],
        content_uri=[
            f"https://ui.adsabs.harvard.edu/link_gateway/{doc['bibcode']}/PUB_HTML",
            f"https://ui.adsabs.harvard.edu/link_gateway/{doc['bibcode']}/EPRINT_PDF",
        ],
        authors=[_normalize_text(x) for x in authors],
        title=_normalize_text(doc["title"][0]),
        categories=[_normalize_text(x) for x in doc.get("keyword", [])],
        abstract=_normalize_text(doc.get("abstract", "")),
        date=doc.get("date", "").split("T")[0],
        doi=doc.get("doi", []),
        publication=doc.get("pub", None),
    )


def _dl_attachment(try_urls: list[str]) -> lf.PDF | str | None:
  mime = magic.Magic(mime=True)
  for url in try_urls:
    if not url:
      continue
    try:
      response = httpx.get(
          url,
          timeout=300,
          follow_redirects=True,
      ).raise_for_status()
    except httpx.HTTPStatusError as exc:
      continue
    if response.status_code == 200:
      mime_type = mime.from_buffer(response.content)
      if mime_type.startswith("application/pdf"):
        return lf.PDF.from_bytes(response.content)
      elif mime_type.startswith("text/html"):
        return response.text
      elif mime_type.startswith("text/plain"):
        return response.text
      raise ValueError(f"Unsupported MIME type: {mime_type}")
  raise ValueError("No valid URLs found")


def _eval_templated_prompt(
    prompt_template: str,
    output_ctor: type[T],
    model: lf.LanguageModel,
    attachments: dict[str, list[str]] | None = None,
    **kwargs,
) -> T:
  attachments = {k: _dl_attachment(v) for k, v in (attachments or {}).items()}
  output = lf.query(
      prompt=prompt_template,
      schema=str,
      lm=model,
      **attachments,
      **kwargs,
  )
  return output_ctor(**_parse_json_response(output))


@click.command()
@click.option("--query", default=None)
@click.option("--limit", type=int, default=None)
@click.option("--database", default=None, type=click.Choice(["arxiv", "ads"]))
@click.option("--has-doi", is_flag=True, default=False)
@click.option("--token", default=None)
def search(
    query: str | None = None,
    limit: int | None = None,
    database: str | None = None,
    has_doi: bool = False,
    token: str | None = None,
) -> None:
  database = database or "arxiv"
  limit = _DEFAULT_SEARCH_LIMIT if limit is None else limit
  match database:
    case "arxiv":
      token = None
      _iter_func = _iter_arxiv_api
    case "ads":
      token = token or os.getenv("ADS_API_KEY")
      _iter_func = lambda **kwargs: _iter_ads_api(token=token, **kwargs)
      if not token:
        raise ValueError("ADS API key is required")
    case _:
      raise ValueError(f"Invalid database: {database}")

  for submission in _iter_func(
      query=query,
      limit=limit,
  ):
    if has_doi and not submission.get("doi"):
      continue
    click.echo(json.dumps(submission, ensure_ascii=False))


@click.command()
@click.option("--query", required=True)
@click.option("--model_name", default=None)
@click.option("--model_key", default=None)
@click.option("--threshold", type=float, default=None)
def relevance(
    query: str,
    model_name: str | None = None,
    model_key: str | None = None,
    threshold: float | None = None,
) -> None:
  model_name = model_name or _DEFAULT_MODEL_NAME
  threshold = threshold or _DEFAULT_THRESHOLD
  lm = lf.LanguageModel.get(model_name, api_key=model_key)

  with open(_RELEVANCE_SCORE_PROMPT_PATH) as f:
    relevance_score_prompt = f.read()

  map_func = lambda record: _eval_templated_prompt(
      prompt_template=relevance_score_prompt,
      output_ctor=RelevanceResponse,
      model=lm,
      query=query,
      submission=record,
  )
  for submission, result, error in lf.concurrent_map(
      map_func,
      _iter_stdin_json(),
      max_workers=8,
      show_progress=True,
  ):
    submission: Submission
    result: RelevanceResponse
    error: Exception | None
    if error:
      raise error
    if result.get("relevance_score", 0) >= threshold:
      combined_output = dict(
          **submission,
          **result,
      )
      click.echo(json.dumps(combined_output, ensure_ascii=False))


@click.command()
@click.option("--model_name", default=None)
@click.option("--model_key", default=None)
@click.option("--threshold", type=float, default=None)
def quality(
    model_name: str | None = None,
    model_key: str | None = None,
    threshold: float | None = None,
) -> None:
  model_name = model_name or _DEFAULT_MODEL_NAME
  threshold = threshold or _DEFAULT_THRESHOLD
  lm = lf.LanguageModel.get(model_name, api_key=model_key)

  with open(_QUALITY_SCORE_PROMPT_PATH) as f:
    qualitative_score_prompt = f.read()

  map_func = lambda record: _eval_templated_prompt(
      prompt_template=qualitative_score_prompt,
      output_ctor=QualityResponse,
      model=lm,
      submission=record,
      attachments=dict(paper=record["content_uri"]),
  )
  for submission, result, error in lf.concurrent_map(
      map_func,
      _iter_stdin_json(),
      max_workers=8,
      show_progress=True,
  ):
    submission: Submission
    result: QualityResponse
    error: Exception | None
    if error:
      click.echo(f"Error: {error}", err=True)
      continue
    if result.get("quality_score", 0) >= threshold:
      combined_output = dict(
          **submission,
          **result,
      )
      click.echo(json.dumps(combined_output, ensure_ascii=False))


# Add commands to the CLI
cli.add_command(search)
cli.add_command(relevance)
cli.add_command(quality)


if __name__ == "__main__":
  cli()
