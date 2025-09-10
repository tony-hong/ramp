# Retrieval-Augmented Modular Prompt Tuning for Low-Resource Data-to-Text Generation
**[Hugging Face Datasets](https://huggingface.co/datasets/tonyhong/ramp)** | **[Gitlab Repository](https://gitlab.com/forfrt/drone/-/tree/main?ref_type=heads)** | **[paper](https://aclanthology.org/2024.lrec-main.1224v2.pdf)**

This is the main repo for the paper Retrieval-Augmented Modular Prompt Tuning for Low-Resource Data-to-Text Generation in LREC-COLING 2024. The original code is in a [Gitlab Repository](https://gitlab.com/forfrt/drone/-/tree/main?ref_type=heads). 

### Abstract
Data-to-text (D2T) generation describes the task of verbalizing data, often given as attribute-value pairs. While this task is relevant for many different data domains beyond the traditionally well-explored tasks of weather forecasting, restaurant recommendations, and sports reporting, a major challenge to the applicability of data-to-text generation methods is typically data sparsity. For many applications, there is extremely little training data in terms of attribute-value inputs and target language outputs available for training a model. Given the sparse data setting, recently developed prompting methods seem most suitable for addressing D2T tasks since they do not require substantial amounts of training data, unlike finetuning approaches. However, prompt-based approaches are also challenging, as a) the design and search of prompts are non-trivial; and b) hallucination problems may occur because of the strong inductive bias of these models. In this paper, we propose a retrieval-augmented modular prompt tuning () method, which constructs prompts that fit the input data closely, thereby bridging the domain gap between the large-scale language model and the structured input data. Experiments show that our method generates texts with few hallucinations and achieves state-of-the-art performance on a dataset for drone handover message generation.

Maintainers: Xudong Hong

Code contributors: Ruitao Feng, Mattes Warning

