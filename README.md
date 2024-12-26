<p align="center" width="100%">
<img src="assets/Mobile-MMLU.png" alt="Owl" style="width: 35%; min-width: 300px; display: block; margin: auto;">
</p>

<p align="center">
    <a href="https://huggingface.co/spaces/MBZUAI-LLM/Mobile-MMLU-Challenge"><img src="https://img.shields.io/badge/%F0%9F%8F%86-leaderboard-blue"></a>
    <a href="https://huggingface.co/datasets/MBZUAI-LLM/Mobile-MMLU"><img src="https://img.shields.io/badge/🤗-dataset-orange"></a>
    <a href="https://vila-lab.github.io/Mobile_MMLU/"><img src="https://img.shields.io/badge/📱-website-8A2BE2"></a>
    <a href="https://github.com/VILA-Lab/Mobile-MMLU/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
    <a href="https://x.com/vila_shen_lab"><img src="https://img.shields.io/twitter/url?url=https%3A%2F%2Fx.com%2Fvila_shen_lab&label=Follow%20%40vila_shen_lab"></a>
    <a href="https://www.python.org/downloads/release/python-3100/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg"></a>
    <a href="https://github.com/VILA-Lab/Mobile-MMLU/issues"><img src="https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat"></a>
</p>

# Mobile-MMLU: A Mobile Intelligence Language Understanding Benchmark

## Overview

Mobile-MMLU is a comprehensive benchmark designed to evaluate mobile-compatible Large Language Models (LLMs) across 80 diverse fields including Education, Healthcare, and Technology. Our benchmark is redefining mobile intelligence evaluation for a smarter future, with a focus on real-world applicability and performance metrics that matter in mobile environments.

## Key Features

- **Comprehensive Coverage**: Spans 80 distinct fields with carefully curated questions
- **Mobile-Optimized**: Specifically designed for evaluating mobile-compatible LLMs
- **16,186 Questions**: Extensive dataset including scenario-based questions
- **Rigorous Evaluation**: Systematic assessment of performance, efficiency, and accuracy
- **Real-world Applications**: Focus on practical use cases in everyday scenarios

## Leaderboard

Visit our [live leaderboard](https://huggingface.co/spaces/SondosMB/Mobile-MMLU) to see the latest performance rankings of various mobile LLMs across different categories and metrics.

## Getting Started

### Backends

We current support the following `backends` for model inference:

* `hf`: [HF Tranformers](https://github.com/huggingface/transformers)
* `gptqmodel`: [GPTQModel](https://github.com/ModelCloud/GPTQModel) for gptq quantized models

### Response Generation

1. Install required packages:
```bash
pip install torch transformers datasets pandas tqdm
```

1. Generate responses using your model:
```bash
python generate_answers.py \
    --model_name your_model_name \
    --batch_size 32 \
    --device cuda
```

The script supports various arguments:
- `--model_name`: Name or path of the model (required)
- `--batch_size`: Batch size for processing (default: 32)
- `--device`: Device to run the model on (default: `auto` = use cuda if available else cpu)
- `--backend`: Load Model on (default: `hf`). Use `gptqmodel` for gptq quantized models.

### Response Format

The script will generate a CSV file with the following format:
```csv
question_id,predicted_answer
q1,A
q2,B
q3,C
...
```

Each row contains:
- `question_id`: The unique identifier for each question
- `predicted_answer`: The model's prediction (A, B, C, or D)

### Submission

1. After generating the CSV file with your model's predictions, submit it through our evaluation portal at [link](https://huggingface.co/spaces/SondosMB/Mobile-MMLU)

