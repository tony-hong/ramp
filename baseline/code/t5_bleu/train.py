import pandas as pd
import numpy as np
import torch
import pyarrow as pa
import pyarrow.dataset as ds
import pandas as pd
from datasets import Dataset
from functools import partial
from datasets import load_dataset, load_metric
from nltk.translate import meteor_score
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

torch.set_num_threads(2)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# load rouge
rouge = load_metric("rouge")
bleu= load_metric("bleu")
meteor = load_metric('meteor')

# load dataset
train_set = pd.read_csv("../../data/all_train.csv")
val_set = pd.read_csv("../../data/all_val.csv")
train_set = train_set.dropna()
val_set = val_set.dropna()

#Prase1
# train_set = train_set.head(100)

### convert to Huggingface dataset
train_set = Dataset.from_pandas(train_set)
val_set = Dataset.from_pandas(val_set)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-base")

# max encoder length for led
encoder_max_length = 512
decoder_max_length = 512
batch_size = 4
gradient_accumulation_steps = 4
noise_lambda = 0
learning_rate = 5e-5
weight_decay = 0.01
num_train_epochs = 100

num_samples = 261
num_steps = float(num_samples) * num_train_epochs / (batch_size * gradient_accumulation_steps)
steps_per_epoch = int(num_steps / num_train_epochs)

def process_data_to_model_inputs(batch):
    # tokenize the inputs and labels
    inputs = tokenizer(
        batch["source"],
        padding="max_length",
        truncation=True,
        max_length=encoder_max_length,
    )
    outputs = tokenizer(
        batch["summary"],
        padding="max_length",
        truncation=True,
        max_length=decoder_max_length,
    )

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask

    # create 0 global_attention_mask lists
    batch["global_attention_mask"] = len(batch["input_ids"]) * [
        [0 for _ in range(len(batch["input_ids"][0]))]
    ]

    # since above lists are references, the following line changes the 0 index for all samples
    batch["global_attention_mask"][0][0] = 1
    batch["labels"] = outputs.input_ids

    # We have to make sure that the PAD token is ignored
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in batch["labels"]
    ]

    return batch

# map train data
train_set = train_set.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size,
    remove_columns=["source", "summary"],
)

# map val data
val_set = val_set.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size,
    remove_columns=["source", "summary"],
)

# set Python list to PyTorch tensor
train_set.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
)

# set Python list to PyTorch tensor
train_set.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
)

# enable fp16 apex training
training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate = learning_rate,
    weight_decay = weight_decay,
    num_train_epochs = num_train_epochs,
    fp16=True,
    fp16_backend="apex",
    output_dir="./",
    logging_steps=steps_per_epoch, 
    eval_steps=steps_per_epoch*5, 
    save_steps=steps_per_epoch*101, 
    warmup_steps=512,
    save_total_limit=2,
    gradient_accumulation_steps = gradient_accumulation_steps, 
    optim= "adafactor"
)


# compute Rouge score during validation
def compute_metrics(pred, tokenizer, rouge, bleu):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    
    pred_tokens=[tokenizer.tokenize(pred) for pred in pred_str]
    refer_tokens=[[tokenizer.tokenize(label)] for label in label_str]

    rouge_output = rouge.compute(predictions=pred_str, references=label_str)
    bleu_output = bleu.compute(predictions=pred_tokens, references=refer_tokens, max_order=4)
    meteor_output = [meteor_score.single_meteor_score(
        tokenizer.tokenize(ref_s), tokenizer.tokenize(pred_s), alpha=0.9, beta=3, gamma=0.5)
                      for ref_s, pred_s in zip(label_str, pred_str)
                     ]
    
    meteor_avg=sum(meteor_output) / len(meteor_output)
    
    avg_score = (rouge_output["rouge1"].mid.fmeasure+rouge_output["rouge2"].mid.fmeasure+
                 rouge_output["rougeL"].mid.fmeasure+rouge_output["rougeLsum"].mid.fmeasure
                  )/4
    
    return {
        "rouge1_precision": round(rouge_output["rouge1"].mid.precision, 4),
        "rouge1_recall": round(rouge_output["rouge1"].mid.recall, 4),
        "rouge1_fmeasure": round(rouge_output["rouge1"].mid.fmeasure, 4),

        "rouge2_precision": round(rouge_output["rouge2"].mid.precision, 4),
        "rouge2_recall": round(rouge_output["rouge2"].mid.recall, 4),
        "rouge2_fmeasure": round(rouge_output["rouge2"].mid.fmeasure, 4),
        
        "rougeL_precision": round(rouge_output["rougeL"].mid.precision, 4),
        "rougeL_recall": round(rouge_output["rougeL"].mid.recall, 4),
        "rougeL_fmeasure": round(rouge_output["rougeL"].mid.fmeasure, 4),
        
        "rougeLsum_precision": round(rouge_output["rougeLsum"].mid.precision, 4),
        "rougeLsum_recall": round(rouge_output["rougeLsum"].mid.recall, 4),
        "rougeLsum_fmeasure": round(rouge_output["rougeLsum"].mid.fmeasure, 4),

        "average_rogue": round(avg_score, 4),
        
        "bleu_score": round(bleu_output["bleu"], 4),
        "bleu-1_score": round(bleu_output["precisions"][0], 4),
        "bleu-2_score": round(bleu_output["precisions"][1], 4),
        "bleu-3_score": round(bleu_output["precisions"][2], 4),
        "bleu-4_score": round(bleu_output["precisions"][3], 4),
        
        "meteor_avg": round(meteor_avg, 4),
    }


# load model + enable gradient checkpointing & disable cache for checkpointing
t5 = AutoModelForSeq2SeqLM.from_pretrained("t5-base", use_cache=False)

#NoisyTune
for name ,para in t5.named_parameters():
    t5.state_dict()[name][:] +=(torch.rand(para.size())-0.5)*noise_lambda*torch.std(para)
    
# #BitFit
# for name ,para in led.named_parameters():
#     if "bias" in name:
#         para.requires_grad = True
#     else:
#         para.requires_grad = False


# set generate hyperparameters
t5.config.num_beams = 2
# ignore the warning message, see https://github.com/huggingface/transformers/issues/5204
t5.config.max_length = 512
t5.config.min_length = 512
t5.config.length_penalty = 2.0
t5.config.early_stopping = True
t5.config.no_repeat_ngram_size = 3

compute_metrics_partial = partial(compute_metrics, tokenizer=tokenizer, rouge=rouge, bleu=bleu)
    
# instantiate trainer
trainer = Seq2SeqTrainer(
    model=t5,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics_partial,
    train_dataset=train_set,
    eval_dataset=val_set,
)


# start training
# torch.autograd.set_detect_anomaly(True)
trainer.train()
trainer.save_model("check/")


# #Evaluation
# # load testset
# test_set = pd.read_csv("../../data/valid_val_df.csv")
# ### convert to Huggingface dataset
# test_set = test_set.dropna()
# test_set = Dataset.from_pandas(test_set)


# # load tokenizer
# model = led.to("cuda").half()

# def generate_answer(batch):
#     inputs_dict = tokenizer(batch["source"], padding="max_length", max_length=8192, return_tensors="pt", truncation=True)
#     input_ids = inputs_dict.input_ids.to("cuda")
#     attention_mask = inputs_dict.attention_mask.to("cuda")
#     global_attention_mask = torch.zeros_like(attention_mask)
#     # put global attention on <s> token
#     global_attention_mask[:, 0] = 1

#     predicted_summary_ids = model.generate(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
#     batch["predicted_summary"] = tokenizer.batch_decode(predicted_summary_ids, skip_special_tokens=True)
#     return batch


# result = test_set.map(generate_answer, batched=True, batch_size=4)
# rouge = load_metric("rouge")

# print("Result:", rouge.compute(predictions=result["predicted_summary"], references=result["summary"], rouge_types=["rouge2"])["rouge2"].mid)

# with open("output_score.txt", "a") as f:
#     print("Result:", rouge.compute(predictions=result["predicted_summary"], references=result["summary"], rouge_types=["rouge2"])["rouge2"].mid, file=f)

# pd.DataFrame(result).to_csv("result.csv")