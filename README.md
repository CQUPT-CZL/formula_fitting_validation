# formula_fitting_validation

A system to evaluate LLM performance in formula fitting tasks.

## Data Format
- `pred.jsonl`: LLM output with `prompt`, `predict`, `label`.
- `test.json`: Raw data with `instruction` and `raw_data` (a, b, c, result).
- `prompt.txt`: Template for generating `calculate_value` function.

    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

## Output File Sharing
- Output files (`metrics_results.json`, plots) are stored in `output/` and tracked by Git LFS.
- To upload new output:
  ```bash
  make evaluate
  git add output/results/
  git commit -m "Update metrics results"
  git push
  git lfs push origin main