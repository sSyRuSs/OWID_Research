# Explainable carbon intensity of national power systems

LaTeX manuscript files:

- `main.tex` — Vietnamese research paper with an English abstract.
- `references.bib` — cited data and methods references.
- `main.pdf` — default compiler output.
- `Explainable_Carbon_Intensity_OWID_2000_2025.pdf` — compiled manuscript.

The figures are read from `../../outputs/thesis/figures/`, so build from this
directory while keeping the project output folder in place:

```powershell
# Standard MiKTeX build
xelatex -interaction=nonstopmode main.tex
bibtex main
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex
```

If MiKTeX cannot initialize its XeLaTeX format (for example, it reports that
the format file is missing), use Tectonic as a standalone fallback rather than
repeatedly retrying the broken format setup:

```powershell
# Run from this paper directory; Tectonic fetches TeX resources as needed.
tectonic --keep-logs main.tex
```

Tectonic handles the bibliography and repeated passes automatically. It creates
`main.pdf`; copy or rename that file to
`Explainable_Carbon_Intensity_OWID_2000_2025.pdf` for the deliverable.
When downloading a standalone Tectonic binary, use an official release and
verify its published SHA-256 checksum before running it. Avoid changing or
reinitializing the user's MiKTeX setup just to compile this paper.

To verify the output, check that the PDF exists and has a `%PDF-` header, then
extract its text with `pdftotext` and confirm the main sections and references
are present.

The reported experiment artifacts are under `../../outputs/thesis/`.
