# Lab Proof

## Workflow
- Trigger: product dataset row from HuggingFace or fallback local sample
- Transform: image encoding + OpenAI Responses API listing generation
- Action: save `output_listings/generated_listings.json` and verify it by reloading the file

## Execution Trace
- Happy-path run: 5 products processed, 0 failures, verified JSON written to [`output_listings/generated_listings.json`](./output_listings/generated_listings.json)
- Refactored run output: see [`product_generator.py`](./product_generator.py) and the committed step `c56c1e3`

## Input Payload
- Example product row comes from the pipeline input dataframe in [`product_generator.py`](./product_generator.py)
- Representative fields: `id`, `name`, `price`, `category`, `image_path`

## Output Record
- Example listing record is the `15970` entry in [`output_listings/generated_listings.json`](./output_listings/generated_listings.json)
- Structure: `metadata` + `generated_listing` with `title`, `description`, `features`, `keywords`

## Before / After
- Before commit: `f864391`
- After commit: `c56c1e3`
- Biggest change: monolithic script -> helper-based pipeline with typed results, explicit errors, and honest verification

## Notes For Reviewer
- Root entry point for the review is this file.
- The script now exits nonzero on prep or verification failure.
- The four-part error format appears in the runtime failure paths inside [`product_generator.py`](./product_generator.py).
