# Retrieval-Augmented Modular Prompt Tuning for Low-Resource Data-to-Text Generation
**[Hugging Face Datasets](https://huggingface.co/datasets/tonyhong/ramp)** | **[Gitlab Repository](https://gitlab.com/forfrt/drone/-/tree/main?ref_type=heads)** | **[paper](https://aclanthology.org/2024.lrec-main.1224v2.pdf)**

> **TL;DR.** RAMP builds input-aware prompts by (1) retrieving semantically similar examples from the training data and (2) routing **modular, attribute-specific prompt tokens** into the final prompt. On a low-resource drone handover data-to-text task, it reduces hallucinations and achieves SOTA results with frozen PLMs. 

This is the main repo for the paper Retrieval-Augmented Modular Prompt Tuning for Low-Resource Data-to-Text Generation in LREC-COLING 2024. The code here is a clean version of the original code, which is easier to follow. The original code used in our work is in a [Gitlab Repository](https://gitlab.com/forfrt/drone/-/tree/main?ref_type=heads). 

---

## Overview
**RAMP (Retrieval-Augmented Modular Prompt Tuning)** targets **low-resource data-to-text (D2T)** generation, where training data is scarce and prompt-only methods are attractive but prone to hallucinations. RAMP:
1) **Retrieves** examples that share the same logical/attribute profile as the input, and  
2) **Routes modular, attribute-conditioned continuous prompts** into the final augmented prompt for a frozen LLM. 

The paper evaluates RAMP on a **drone handover message generation** dataset and reports state-of-the-art results with fewer hallucinations. 

---

## What’s in this repo
- `code/` – Notebooks and/or scripts to train/run RAMP with T5/Flan-T5/LED.
- `data/` – Dataset (see **Data**).
- `docs/` – Presentations and documents.
- `baseline/` - Baselines trained end-to-end. 
- `FastChat/` – Vicuna baselines. 

---

## Data

The experiments use a **low-resource Drone Handover** D2T dataset consisting of ~1.6K examples of sensor/telemetry records and human-readable handover messages, with ~25 tracked attributes (e.g., altitude, speed, battery). Splits follow prior work.

You can load a processed version via Hugging Face Datasets:

```python
from datasets import load_dataset
ds = load_dataset("tonyhong/ramp")  # provides train/validation/test if available
```

---

## Evaluation

We report the following automatic metrics:
- **BLEU**, **ROUGE**, **METEOR**, and **PARENT** (for table-to-text factuality).

Example with 🤗 Evaluate:

```python
import evaluate

bleu = evaluate.load("sacrebleu")
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")

bleu_res = bleu.compute(predictions=preds, references=[[r] for r in refs])
rouge_res = rouge.compute(predictions=preds, references=refs)
meteor_res = meteor.compute(predictions=preds, references=refs)

print(bleu_res, rouge_res, meteor_res)
```

For **PARENT**, use an available implementation.

---

## Results (from the paper)

On the drone handover test set, RAMP achieves large gains over zero/one-shot prompting and fixed/retrieved example prompts. Selected results:

| Model (frozen PLM) | BLEU | ROUGE | METEOR | PARENT |
|---|---:|---:|---:|---:|
| Vicuna-13B 0-shot | 1.34 | 14.58 | 21.99 | 8.72 |
| Vicuna-13B 1-shot | 17.88 | 23.98 | 34.87 | 11.51 |
| T5 + RAMP         | 78.92 | 89.82 | 90.30 | 67.45 |
| Flan-T5 + RAMP    | 85.12 | 92.37 | 92.48 | 70.99 |
| LED + RAMP        | **91.76** | **96.07** | **94.92** | **74.92** |

> See the paper for full ablations (fixed vs. retrieved examples vs. RAMP) and human evaluation.


---

## Cite

If you use this code or ideas, please cite:

```bibtex
@inproceedings{feng-etal-2024-retrieval,
  title     = {Retrieval-Augmented Modular Prompt Tuning for Low-Resource Data-to-Text Generation},
  author    = {Feng, Ruitao and Hong, Xudong and Jobanputra, Mayank and Warning, Mattes and Demberg, Vera},
  booktitle = {Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)},
  year      = {2024},
  month     = {May},
  address   = {Torino, Italia},
  pages     = {14053--14062},
  publisher = {ELRA and ICCL},
  url       = {https://aclanthology.org/2024.lrec-main.1224/}
}
```

## Paper Abstract
Data-to-text (D2T) generation describes the task of verbalizing data, often given as attribute-value pairs. While this task is relevant for many different data domains beyond the traditionally well-explored tasks of weather forecasting, restaurant recommendations, and sports reporting, a major challenge to the applicability of data-to-text generation methods is typically data sparsity. For many applications, there is extremely little training data in terms of attribute-value inputs and target language outputs available for training a model. Given the sparse data setting, recently developed prompting methods seem most suitable for addressing D2T tasks since they do not require substantial amounts of training data, unlike finetuning approaches. However, prompt-based approaches are also challenging, as a) the design and search of prompts are non-trivial; and b) hallucination problems may occur because of the strong inductive bias of these models. In this paper, we propose a retrieval-augmented modular prompt tuning () method, which constructs prompts that fit the input data closely, thereby bridging the domain gap between the large-scale language model and the structured input data. Experiments show that our method generates texts with few hallucinations and achieves state-of-the-art performance on a dataset for drone handover message generation.

## License
This project is licensed under the **Apache License 2.0**. See [LICENSE](./LICENSE).

---

## Contact
Questions or issues? Please open a GitHub issue. For paper-related questions, see the author list and contact details in the publication.

Maintainers: Xudong Hong

Code contributors: Ruitao Feng, Mattes Warning

---

## Acknowledgments
- The dataset and task setup originate from [prior work](https://aclanthology.org/2022.lrec-1.745.pdf) (Chang et al., 2022); see related references in the paper.
- [Repo of Chang et al., 2022](https://gitlab.com/erniecyc/drone/)
