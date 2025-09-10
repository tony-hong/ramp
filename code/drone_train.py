from logic.description_logic import *
from utils.load_sensor_data import *
from data_augmentation.text_augmentation import *
from utils.table_text_eval import *

import json
import math
import torch

import pandas as pd
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from datasets import load_metric
from nltk.translate import meteor_score
from sklearn.metrics import accuracy_score
from transformers import T5Tokenizer, T5ForConditionalGeneration


# old_format_example=pd.ExcelFile(r"../text_annotation/rural.xlsx")

dataset_file_li=[r"../../text_annotation/desert.xlsx", r"../../text_annotation/disturbance.xlsx", r"../../text_annotation/factory.xlsx", r"../../text_annotation/island.xlsx", 
                 r"../../text_annotation/misc.xlsx", r"../../text_annotation/ocean.xlsx", r"../../text_annotation/rural.xlsx", r"../../text_annotation/urban.xlsx"
]

# dataset_file_li=[ r"../text_annotation/rural.xlsx"]

def load_sensor_data(dataset_file_li=dataset_file_li):
    old_data_dic=dict()

    for dataset_file in dataset_file_li:
        # print("----processing file: {}----".format(dataset_file))

        old_format_example=pd.ExcelFile(dataset_file)

        ts_converters={'name': str, 'Type': str, 'Moving': str2bool, 'InPath': str2bool, 'time_stamp': str}
        status_converters={'PilotExperienced': str2bool, 'Low_visibility': str2bool, 'Normal_frame': str2bool,
                          'weather': str, 'upside_down': str2bool, 'good_motor_condition': str2bool,
                          'going_backwards': str2bool, 'indoor': str2bool, 'waterproof drone?': str2bool,
                          'flying over': str, 'criticality': str, 
                          'RiskOfPhysicalDamage': str, 'RiskOfInternalDamage': str2bool,
                          'RiskOfHumanDamage': str2bool, 'LostConnection': str2bool}
        text_converters={'Link':str, 'Text1':str, 'Text2':str}

        old_format_ts=pd.read_excel(old_format_example, sheet_name='timestep', converters=ts_converters)
        old_format_status=pd.read_excel(old_format_example, sheet_name='status', converters=status_converters)
        old_format_text=pd.read_excel(old_format_example, sheet_name='text', converters=text_converters)

        # print("---loading sensor data---")
        load_ts_to_dict(old_format_ts, old_data_dic)
        load_status_to_dict(old_format_status, old_data_dic)
        load_text_to_dict(old_format_text, old_data_dic)

        # print("---generating templates---")
        for link, data in old_data_dic.items():
            templates, related_status_dic, related_timestep_dic = gen_templates(data)
            old_data_dic[link]['templates'] = templates
            old_data_dic[link]['related_status_dic'] = related_status_dic
            old_data_dic[link]['related_timestep_dic'] = related_timestep_dic
            
    return old_data_dic

# text_li, all_status_dic_li, all_timestep_dic_li, all_sensor_data_li have the length of all the label text. eg, 300.
# templates_li, related_sensor_data_li have the length of all the filtered sentences. eg, 1652.
def read_data_dict(old_data_dic):
    
    link_li=[]
    text_li=[]
    dl_name_li=[]
    all_status_dic_li=[]
    all_timestep_dic_li=[]
    all_sensor_data_li=[]
    
    templates_li=[]
    related_status_dic_li=[]
    related_timestep_dic_li=[]
    related_sensor_data_li=[]
    dl_related_sensor_data_mapping_li=[]
    
    oneshot_examples_li=[]

    # related_sensor_data_li as input and templates_li as label
    for index, (link, value) in enumerate(old_data_dic.items()):
        link_li.append(link)
        
        text_li.append(value.get('text', []))
        status_tmp=value.get('status', [])
        timestep_tmp=value.get('timestep', [])
        
        all_status_dic_li.append(status_tmp)
        all_timestep_dic_li.append(timestep_tmp)
        
        all_sensor_data_li.append({'status': status_tmp, 'timestep': timestep_tmp})
            
        dl_index=0

        for dl_name, template_value in value['templates'].items():
            templates_li.extend(template_value)
            if dl_name in value['related_status_dic']:
                related_status_dic_li.extend(value['related_status_dic'][dl_name])
                related_sensor_data_li.extend(value['related_status_dic'][dl_name])
                for i in range(len(value['related_status_dic'][dl_name])):
                    oneshot_examples_li.append(WARNING_SITUATION_EXAMPLES[dl_name])
            elif dl_name in value['related_timestep_dic']:
                related_timestep_dic_li.extend(value['related_timestep_dic'][dl_name])
                related_sensor_data_li.extend(value['related_timestep_dic'][dl_name])
                for i in range(len(value['related_timestep_dic'][dl_name])):
                    oneshot_examples_li.append(WARNING_SITUATION_EXAMPLES[dl_name])
            else:
                print("WHAT?")

    related_sensor_data_li=[json.dumps(dic) for dic in related_sensor_data_li]
    
    return (text_li, all_status_dic_li, all_timestep_dic_li, all_sensor_data_li), (templates_li, related_sensor_data_li), oneshot_examples_li



def train_test_split(all_li, train_ratio=0.6):
    all_len=len(all_li)
    train_len=math.ceil(train_ratio*all_len)
    
    return all_li[:train_len], all_li[train_len:]


def generate_data(batch_size, n_tokens, related_sensor_data, text_data, tokenizer, device):

    def yield_data(x_batch, y_batch, l_batch):
            
        x = torch.nn.utils.rnn.pad_sequence(x_batch, batch_first=True)
        y = torch.nn.utils.rnn.pad_sequence(y_batch, batch_first=True)
#         y = torch.cat(y_batch, dim=0)
            
        # Todo: Add prompt mask for different description logic
        m = (x > 0).to(torch.float32)
        decoder_input_ids = torch.full((x.shape[0], y.shape[1]), 1)
        
        if torch.cuda.is_available():
            x = x.to(device)
            y = y.to(device)
            m = m.to(device)
            decoder_input_ids = decoder_input_ids.to(device)
        return x, y, m, decoder_input_ids, l_batch

    x_batch, y_batch, l_batch = [], [], []
    for x, y in zip(related_sensor_data, text_data):
        
        context = x
        inputs = tokenizer(context, return_tensors="pt")
        inputs['input_ids'] = torch.cat([torch.full((1, n_tokens), 1), inputs['input_ids']], 1)
        
        outputs=tokenizer(y, return_tensors="pt")
#         print(type(inputs), inputs['input_ids'][0].size(), inputs['input_ids'].size(), 
#               type(outputs), outputs['input_ids'][0].size(), outputs['input_ids'].size())
        
        outputs_ids = torch.cat([torch.full((1, n_tokens - 1), -100), outputs['input_ids']], 1)
        
        x_batch.append(inputs['input_ids'][0])
        y_batch.append(outputs_ids[0])
        l_batch.append(outputs['input_ids'][0])
        if len(x_batch) >= batch_size:
            yield yield_data(x_batch, y_batch, l_batch)
            x_batch, y_batch, l_batch = [], [], []

    if len(x_batch) > 0:
        yield yield_data(x_batch, y_batch, l_batch)
        x_batch, y_batch, l_batch = [], [], []


def prefix_one_shot(input_str, one_shot_input, one_shot_output):
    return "input: {one_shot_input}, output: {one_shot_output}\n input:{input_str}, output:".format(
        one_shot_input=one_shot_input, one_shot_output=one_shot_output, input_str=input_str
    )


def generate_oneshot_data(batch_size, n_tokens, related_sensor_data, text_data, oneshot_data):

    def yield_data(x_batch, y_batch, l_batch):
            
        x = torch.nn.utils.rnn.pad_sequence(x_batch, batch_first=True)
        y = torch.nn.utils.rnn.pad_sequence(y_batch, batch_first=True)
#         y = torch.cat(y_batch, dim=0)
            
        # Todo: Add prompt mask for different description logic
        m = (x > 0).to(torch.float32)
        decoder_input_ids = torch.full((x.shape[0], y.shape[1]), 1)
        
        if torch.cuda.is_available():
            x = x.to(device)
            y = y.to(device)
            m = m.to(device)
            decoder_input_ids = decoder_input_ids.to(device)
        return x, y, m, decoder_input_ids, l_batch

    x_batch, y_batch, l_batch = [], [], []
    for x, y, oneshot in zip(related_sensor_data, text_data, oneshot_data):
        oneshot_input, oneshot_output=oneshot
        context=prefix_one_shot(x, oneshot_input, oneshot_output)
        inputs = tokenizer(context, return_tensors="pt")
        inputs['input_ids'] = torch.cat([torch.full((1, n_tokens), 1), inputs['input_ids']], 1)
        
        outputs=tokenizer(y, return_tensors="pt")
#         print(type(inputs), inputs['input_ids'][0].size(), inputs['input_ids'].size(), 
#               type(outputs), outputs['input_ids'][0].size(), outputs['input_ids'].size())
        
        outputs_ids = torch.cat([torch.full((1, n_tokens - 1), -100), outputs['input_ids']], 1)
        
        x_batch.append(inputs['input_ids'][0])
        y_batch.append(outputs_ids[0])
        l_batch.append(outputs['input_ids'][0])
        if len(x_batch) >= batch_size:
            yield yield_data(x_batch, y_batch, l_batch)
            x_batch, y_batch, l_batch = [], [], []

    if len(x_batch) > 0:
        yield yield_data(x_batch, y_batch, l_batch)
        x_batch, y_batch, l_batch = [], [], []
        
        
def generate_augmented_data(batch_size, n_tokens, related_sensor_data, text_data, tokenizer, device):

    def yield_data(x_batch, y_batch, l_batch):
            
        x = torch.nn.utils.rnn.pad_sequence(x_batch, batch_first=True)
        y = torch.nn.utils.rnn.pad_sequence(y_batch, batch_first=True)
#         y = torch.cat(y_batch, dim=0)
            
        # Todo: Add prompt mask for different description logic
        m = (x > 0).to(torch.float32)
        decoder_input_ids = torch.full((x.shape[0], y.shape[1]), 1)
        
        if torch.cuda.is_available():
            x = x.to(device)
            y = y.to(device)
            m = m.to(device)
            decoder_input_ids = decoder_input_ids.to(device)
        return x, y, m, decoder_input_ids, l_batch

    x_batch, y_batch, l_batch = [], [], []
    for x, y in zip(related_sensor_data, text_data):
        
        context = x
        inputs = tokenizer(context, return_tensors="pt")
        inputs['input_ids'] = torch.cat([torch.full((1, n_tokens), 1), inputs['input_ids']], 1)
        
        outputs=tokenizer(y, return_tensors="pt")
#         print(type(inputs), inputs['input_ids'][0].size(), inputs['input_ids'].size(), 
#               type(outputs), outputs['input_ids'][0].size(), outputs['input_ids'].size())
        
        outputs_ids = torch.cat([torch.full((1, n_tokens - 1), -100), outputs['input_ids']], 1)
        
        x_batch.append(inputs['input_ids'][0])
        y_batch.append(outputs_ids[0])
        l_batch.append(outputs['input_ids'][0])
        if len(x_batch) >= batch_size:
            yield yield_data(x_batch, y_batch, l_batch)
            x_batch, y_batch, l_batch = [], [], []

    if len(x_batch) > 0:
        yield yield_data(x_batch, y_batch, l_batch)
        x_batch, y_batch, l_batch = [], [], []


class SoftEmbedding(nn.Module):
    def __init__(self, 
                wte: nn.Embedding,
                n_tokens: int = 10, 
                random_range: float = 0.5,
                initialize_from_vocab: bool = True):
        """appends learned embedding to 
        Args:
            wte (nn.Embedding): original transformer word embedding
            n_tokens (int, optional): number of tokens for task. Defaults to 10.
            random_range (float, optional): range to init embedding (if not initialize from vocab). Defaults to 0.5.
            initialize_from_vocab (bool, optional): initalizes from default vocab. Defaults to True.
        """
        super(SoftEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.learned_embedding = nn.parameter.Parameter(self.initialize_embedding(wte,
                                                                               n_tokens, 
                                                                               random_range, 
                                                                               initialize_from_vocab))
            
    def initialize_embedding(self, 
                             wte: nn.Embedding,
                             n_tokens: int = 10, 
                             random_range: float = 0.5, 
                             initialize_from_vocab: bool = True):
        """initializes learned embedding
        Args:
            same as __init__
        Returns:
            torch.float: initialized using original schemes
        """
        if initialize_from_vocab:
            return self.wte.weight[:n_tokens].clone().detach()
        return torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(-random_range, random_range)
            
    def forward(self, tokens):
        """run forward pass
        Args:
            tokens (torch.long): input tokens before encoding
        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        input_embedding = self.wte(tokens[:, self.n_tokens:])
        learned_embedding = self.learned_embedding.repeat(input_embedding.size(0), 1, 1)
        return torch.cat([learned_embedding, input_embedding], 1)
    

def train_with_prompt(model, tokenizer, device, n_epoch, batch_size, n_tokens,
                      train_input, train_label, val_input, val_label, optimizer, ce_loss, use_ce_loss=False):
    
    total_batch = math.ceil(len(train_input) / batch_size)
    dev_total_batch = math.ceil(len(val_input) / batch_size)
    
    test_losses_epoch=[]
    bleu_score_epoch=[]
    rouge_score_epoch=[]
    meteor_score_epoch=[]

    bleu=load_metric("bleu")
    rouge=load_metric("rouge")
    meteor=load_metric('meteor')

    for epoch in range(n_epoch):
        print('epoch', epoch)

        all_true_labels = []
        all_pred_labels = []
        train_losses = []
        pbar = tqdm(enumerate(generate_data(batch_size, n_tokens, train_input, train_label, tokenizer, device)), total=total_batch)
        for i, (x, y, m, dii, true_labels) in pbar:
            all_true_labels += true_labels

            optimizer.zero_grad()
            outputs = model(input_ids=x, labels=y, attention_mask=m, decoder_input_ids=dii)
            pred_labels = outputs['logits'][:, :, :].argmax(-1).detach().cpu().numpy().tolist()
            all_pred_labels += pred_labels

            if use_ce_loss:
                logits = outputs['logits'][:, :, :]
                true_labels_tensor = torch.tensor(true_labels, dtype=torch.long).cuda()
                loss = ce_loss(logits, true_labels_tensor)
            else:
                loss = outputs.loss
                
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu().numpy().tolist()) / batch_size
            train_losses.append(loss_value)

    #         acc = accuracy_score(all_true_labels, all_pred_labels)
            pbar.set_description(f'train: loss={np.mean(train_losses):.4f}')

        all_true_labels = []
        all_pred_labels = []
        output_max_pred=[]
        test_losses = []

        with torch.no_grad():
            pbar = tqdm(enumerate(generate_data(batch_size, n_tokens, val_input, val_label, tokenizer, device)), total=dev_total_batch)
            for i, (x, y, m, dii, true_labels) in pbar:
                
                all_true_labels += true_labels
                outputs = model(input_ids=x, labels=y, attention_mask=m, decoder_input_ids=dii)

                output_ids=outputs.logits[:, n_tokens:, :]
                max_pred_ids=outputs.logits[:, n_tokens:, :].argmax(-1).detach().cpu().numpy().tolist()
                
                for i, (max_pred_ids_unbatch, output_ids_unbatch) in enumerate(zip(max_pred_ids, output_ids)):
                    max_pred_ids[i]=max_pred_ids_unbatch[max_pred_ids_unbatch!=0]
                    output_ids[i]=output_ids_unbatch[max_pred_ids_unbatch!=0]

                output_max_pred.extend(tokenizer.batch_decode(max_pred_ids))

                loss = outputs.loss
                loss_value = float(loss.detach().cpu().numpy().tolist()) / batch_size
                test_losses.append(loss_value)

                pred_labels = outputs['logits'][:, :, :].argmax(-1).detach().cpu().numpy().tolist()
                all_pred_labels += pred_labels
                pbar.set_description(f'dev: loss={np.mean(test_losses):.4f}')

        bleu_preds=[tokenizer.tokenize(pred) for pred in output_max_pred]
        bleu_refers=[[tokenizer.tokenize(label)] for label in val_label]

        meteor_output = [meteor_score.single_meteor_score(
            tokenizer.tokenize(ref_s), tokenizer.tokenize(pred_s), alpha=0.9, beta=3, gamma=0.5)
                         for ref_s, pred_s in zip(output_max_pred, val_label)
                        ]
        meteor_avg_score=sum(meteor_output) / len(meteor_output)
    
        print("meteor_avg_score", meteor_avg_score)
        bleu_score=bleu.compute(predictions=bleu_preds, references=bleu_refers)
        rouge_score=rouge.compute(predictions=output_max_pred, references=val_label)

        bleu_score_epoch.append(bleu_score)
        rouge_score_epoch.append(rouge_score)
        meteor_score_epoch.append(meteor_avg_score)

        test_losses_epoch.append(np.mean(test_losses))
                  
    return test_losses_epoch, bleu_score_epoch, rouge_score_epoch, meteor_score_epoch


def eval_with_prompt(model, tokenizer, device, batch_size, n_tokens,
                     val_input, val_label, optimizer, ce_loss, use_ce_loss=False):
    dev_total_batch = math.ceil(len(val_input) / batch_size)
    
    all_true_labels = []
    all_pred_labels = []
    output_max_pred=[]
    test_losses = []
        
    with torch.no_grad():
            pbar = tqdm(enumerate(generate_data(batch_size, n_tokens, val_input, val_label, tokenizer, device)), total=dev_total_batch)
            for i, (x, y, m, dii, true_labels) in pbar:
                
                all_true_labels += true_labels
                outputs = model(input_ids=x, labels=y, attention_mask=m, decoder_input_ids=dii)

                output_ids=outputs.logits[:, n_tokens:, :]
                max_pred_ids=outputs.logits[:, n_tokens:, :].argmax(-1).detach().cpu().numpy().tolist()
                
                for i, (max_pred_ids_unbatch, output_ids_unbatch) in enumerate(zip(max_pred_ids, output_ids)):
                    max_pred_ids[i]=max_pred_ids_unbatch[max_pred_ids_unbatch!=0]
                    output_ids[i]=output_ids_unbatch[max_pred_ids_unbatch!=0]
                
#                 print("tokenizer.batch_decode(max_pred_ids)", tokenizer.batch_decode(max_pred_ids))
                output_max_pred.extend(tokenizer.batch_decode(max_pred_ids))

                loss = outputs.loss
                loss_value = float(loss.detach().cpu().numpy().tolist()) / batch_size
                test_losses.append(loss_value)

                pred_labels = outputs['logits'][:, :, :].argmax(-1).detach().cpu().numpy().tolist()
                all_pred_labels += pred_labels
                pbar.set_description(f'dev: loss={np.mean(test_losses):.4f}')
                
    return output_max_pred
                
def save_metrics(test_losses_epoch, bleu_score_epoch, rouge_score_epoch, meteor_score_epoch, metrics_file='metrics_epoch.csv'):
    rouge1_precision_epoch=[round(rouge_output["rouge1"].mid.precision, 4) for rouge_output in rouge_score_epoch]
    rouge1_recall_epoch=[round(rouge_output["rouge1"].mid.recall, 4) for rouge_output in rouge_score_epoch]
    rouge1_fmeasure_epoch=[round(rouge_output["rouge1"].mid.fmeasure, 4) for rouge_output in rouge_score_epoch]

    rouge2_precision_epoch=[round(rouge_output["rouge2"].mid.precision, 4) for rouge_output in rouge_score_epoch]
    rouge2_recall_epoch=[round(rouge_output["rouge2"].mid.recall, 4) for rouge_output in rouge_score_epoch]
    rouge2_fmeasure_epoch=[round(rouge_output["rouge2"].mid.fmeasure, 4) for rouge_output in rouge_score_epoch]

    rougeL_precision_epoch=[round(rouge_output["rougeL"].mid.precision, 4) for rouge_output in rouge_score_epoch]
    rougeL_recall_epoch=[round(rouge_output["rougeL"].mid.recall, 4) for rouge_output in rouge_score_epoch]
    rougeL_fmeasure_epoch=[round(rouge_output["rougeL"].mid.fmeasure, 4) for rouge_output in rouge_score_epoch]

    rougeLsum_precision_epoch=[round(rouge_output["rougeLsum"].mid.precision, 4) for rouge_output in rouge_score_epoch]
    rougeLsum_recall_epoch=[round(rouge_output["rougeLsum"].mid.recall, 4) for rouge_output in rouge_score_epoch]
    rougeLsum_fmeasure_epoch=[round(rouge_output["rougeLsum"].mid.fmeasure, 4) for rouge_output in rouge_score_epoch]


    bleu_avg_score_epoch= [round(bleu_output["bleu"], 4) for bleu_output in bleu_score_epoch]
    bleu_1_score_epoch=[round(bleu_output["precisions"][0], 4) for bleu_output in bleu_score_epoch]
    bleu_2_score_epoch=[round(bleu_output["precisions"][1], 4) for bleu_output in bleu_score_epoch]
    bleu_3_score_epoch=[round(bleu_output["precisions"][2], 4) for bleu_output in bleu_score_epoch]
    bleu_4_score_epoch=[round(bleu_output["precisions"][3], 4) for bleu_output in bleu_score_epoch]

    meteor_score_epoch=[round(meteor_avg, 4) for meteor_avg in meteor_score_epoch]

    df = pd.DataFrame({
        'rouge1_precision_epoch': rouge1_precision_epoch,
        'rouge1_recall_epoch': rouge1_recall_epoch,
        'rouge1_fmeasure_epoch': rouge1_fmeasure_epoch,

        'rouge2_precision_epoch': rouge2_precision_epoch,
        'rouge2_recall_epoch': rouge2_recall_epoch,
        'rouge2_fmeasure_epoch': rouge2_fmeasure_epoch,

        'rougeL_precision_epoch': rougeL_precision_epoch,
        'rougeL_recall_epoch': rougeL_recall_epoch,
        'rougeL_fmeasure_epoch': rougeL_fmeasure_epoch,

        'rougeLsum_precision_epoch': rougeLsum_precision_epoch,
        'rougeLsum_recall_epoch': rougeLsum_recall_epoch,
        'rougeLsum_fmeasure_epoch': rougeLsum_fmeasure_epoch,

        'bleu_avg_score_epoch': bleu_avg_score_epoch,
        'bleu_1_score_epoch': bleu_1_score_epoch,
        'bleu_2_score_epoch': bleu_2_score_epoch,
        'bleu_3_score_epoch': bleu_3_score_epoch,
        'bleu_4_score_epoch': bleu_4_score_epoch,

        'meteor_score_epoch': meteor_score_epoch,
    })

    df.to_csv(metrics_file, index=False)
        
    return (rouge1_precision_epoch, rouge1_recall_epoch, rouge1_fmeasure_epoch), (rouge1_precision_epoch, rouge1_recall_epoch, rouge1_fmeasure_epoch), (rougeL_precision_epoch, rougeL_recall_epoch, rougeL_fmeasure_epoch), (rougeLsum_precision_epoch, rougeLsum_recall_epoch, rougeLsum_fmeasure_epoch), (bleu_avg_score_epoch, bleu_1_score_epoch, bleu_2_score_epoch, bleu_3_score_epoch, bleu_4_score_epoch), (meteor_score_epoch)
    