# Reading materials

Curated papers and notes in a **single Markdown file per work**. Use this folder while iterating on `train.py`: add new references when you discover them, keep summaries honest, and prefer links over pasting copyrighted PDFs verbatim.

## File naming

- `NNNN-short-slug.md` — four-digit sequence (0001, 0002, …) + ASCII slug. Increment the highest existing number when adding a file.
- Or keep a stable slug without sequence if you prefer; the number helps sort chronologically of *addition* to this repo.

## Required structure

Each paper file has two parts:

### 1. YAML front matter (between `---` lines)

Machine-friendly metadata. All keys are optional except **title** and **year** (recommended minimum).

| Key | Description |
|-----|-------------|
| `id` | Stable id, e.g. `chandola-2009-survey` |
| `type` | `paper` \| `preprint` \| `book` \| `thesis` \| `note` |
| `title` | Full title |
| `authors` | String or list; e.g. `"Chandola, V. et al."` |
| `year` | Four-digit year |
| `venue` | Journal, conference, or publisher |
| `doi` | DOI if available |
| `url` | Canonical link (DOI resolver, publisher, or arXiv abstract page) |
| `arxiv` | arXiv id if applicable, e.g. `2301.12345` |
| `tags` | List of short labels, e.g. `[survey, anomaly-detection]` |
| `added` | ISO date the file was added here, e.g. `2026-03-26` |
| `source` | How it entered the folder: `manual`, `arxiv`, `agent`, etc. |

### 2. Markdown body (after the closing `---`)

Standard headings (adapt as needed):

```markdown
# {title}

## Abstract
Short abstract in your own words or a quoted excerpt **only if** usage is allowed.

## Summary
Bullet points: problem, method, relevance to this benchmark.

## Full text (optional)
- **Prefer:** link only (`url` in front matter).
- If you have **open-access** full text or **your own** Markdown conversion, you may place it here. Do not paste paywalled or copyrighted material without permission.

## BibTeX (optional)
```

Copy `_TEMPLATE.md` when adding a new entry.

## Pipeline integration (autoresearch)

- At the **start** of every `train.py` run, `prepare.reading_materials_snapshot()` scans this directory (excluding `README.md` and `_TEMPLATE.md`) and parses scalar fields from each file’s YAML front matter (`id`, `title`, `year`, `url`, etc.).
- That snapshot is written to **`reading_materials_snapshot.json`** under `--results-root` and attached to each line in **`experiment_log.jsonl`** as the `reading_materials` field, alongside `primary_metric_mean` and run metadata.
- Stdout includes `reading_materials_count=N` and the path to the JSON snapshot so agents can grep logs without opening every markdown file.

## For coding agents

- **Read** existing files before proposing changes grounded in literature.
- **Add** new `NNNN-*.md` files when you find a relevant paper; fill front matter completely; write `Summary` yourself—do not invent citations.
- **Do not** edit `prepare.py`; the indexer lives there so every run records which papers were in the repo at experiment time.
