# Earlier carbon-intensity ML paper

- Source: `carbon_intensity_ml_owid_2000_2025.tex`
- Compiled paper: `carbon_intensity_ml_owid_2000_2025.pdf`
- Generated LaTeX tables/macros: `generated_*.tex`
- Figures: `../../outputs/carbon_intensity/figures/`

The project workflow regenerates the paper-specific LaTeX tables/macros in this
folder using `src/carbon_intensity_workflow.py`.

Compile from this directory with XeLaTeX:

```powershell
xelatex -interaction=nonstopmode -halt-on-error carbon_intensity_ml_owid_2000_2025.tex
xelatex -interaction=nonstopmode -halt-on-error carbon_intensity_ml_owid_2000_2025.tex
```
