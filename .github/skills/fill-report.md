# Fill Report

Fill in the LaTeX report for the 02460 Advanced Machine Learning mini-project while respecting template constraints.

## Process

1. **Read project requirements** from `02460_project1.pdf` to understand what needs to be reported

2. **Read the template** at `reports/template.tex` to extract:
   - Character limits (each `\lipsum[X]` ≈ 450-500 characters - NEVER exceed)
   - Image dimensions (e.g., `2.5cm x 2.5cm`, `1cm x 1cm`) - maintain exactly
   - Table structures and column counts
   - Two-column layout constraints

3. **Read current report** at `reports/report.tex` to see what's filled vs placeholder

4. **Scan workspace for data**:
   - Part A: `src/part_a_vaes/`, `reports/outputs/part_a_vaes/`
   - Part B: `src/part_b_diffusion/`, `reports/outputs/part_b_diffusion/fid/*.json`

5. **Copy images before using**: Any image from `reports/outputs/` must be copied to `reports/report_images/` first

6. **Write content** respecting character limits and image dimensions exactly

## Image Rules

- Copy from `reports/outputs/` → `reports/report_images/` before referencing
- Replace `\rule{W}{H}` with `\includegraphics[width=W,height=H]{report_images/filename.png}`
- Keep exact dimensions from template

## Key Files

| Purpose | Location |
|---------|----------|
| Report | `reports/report.tex` |
| Template | `reports/template.tex` |
| Report images | `reports/report_images/` |
| VAE outputs | `reports/outputs/part_a_vaes/` |
| DDPM outputs | `reports/outputs/part_b_diffusion/` |
| FID scores | `reports/outputs/part_b_diffusion/fid/*.json` |
