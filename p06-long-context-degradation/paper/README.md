# Paper — P6: Disentangling Token Count and Position Under Perfect Retrieval in Long Contexts

NeurIPS-style LaTeX draft for Overleaf upload.

## Files

| File | Purpose |
|---|---|
| `main.tex` | Main manuscript (abstract, intro, related work, method, experiments) |
| `neurips_2024.sty` | NeurIPS 2024 style (preprint mode) |
| `references.bib` | Bibliography seed entries |

## Overleaf

1. Zip this folder: `cd paper && zip -r ../p06-long-context-degradation-paper.zip .`
2. In Overleaf: **New Project → Upload Project** → select the zip.
3. Set `main.tex` as the main document.
4. Compiler: **pdfLaTeX** (or LaTeX + pdfLaTeX).

For camera-ready, change `\usepackage[preprint]{neurips_2024}` to `\usepackage[final]{neurips_2024}`.

## Sync with code

- Research foundation: `../research.md` → `common/research/`
- Experiments: `../run.md`, `../results/`
- Update `\section{Results}` once metrics exist.

## Build locally

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
