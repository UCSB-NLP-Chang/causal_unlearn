import json
import pickle
import random
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import datasets
import stanza
import numpy as np

from utils import get_model_identifiers_from_yaml, split_document, replace_name, add_dataset_index, unlearn_prompt


def convert_raw_data_to_model_format(tokenizer, max_length, question, answer, model_configs, document=None, prompt_unlearn=False, unlearn_targets=[]):
    if document is None:
        question_start_token, question_end_token, answer_token = model_configs['question_start_tag'], model_configs['question_end_tag'], model_configs['answer_tag']
        if prompt_unlearn:
            assert len(unlearn_targets) > 0
            new_question = unlearn_prompt.format(entity=unlearn_targets, question=question)
        else:
            new_question = question_start_token + question + question_end_token
        new_answer = answer_token + answer
        full_text = new_question + new_answer
        num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))
    else:
        full_text = document
        num_question_tokens = 0

    encoded = tokenizer(
        full_text, 
        add_special_tokens=True, 
        max_length=max_length, 
        truncation=True
    )
    pad_length = max_length - len(encoded.input_ids)
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)

    # change label to -100 for question tokens
    for i in range(num_question_tokens): label[i] = -100

    return torch.tensor(pad_input_ids), torch.tensor(label), torch.tensor(pad_attention_mask)


class TextForgetDatasetQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family,  max_length=512, split = "forget10", loss_type="npo", input_type="question"):
        super(TextForgetDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.forget_data = datasets.load_dataset(data_path, split)["train"]
        if 'TOFU' in data_path:
            retain_split = "retain" + str(100 - int(split.replace("forget", ""))).zfill(2)
        else:
            retain_split = "retain"
        print('='*20 + f"Loading from {retain_split}" + '='*20)
        self.retain_data =datasets.load_dataset(data_path, retain_split)["train"]
        if 'TOFU' in data_path:
            # make sure train and test sets do not overlap
            retain_eval = datasets.load_dataset(data_path, 'retain_perturbed')["train"]
            eval_questions = {i['question'] for i in retain_eval}
            keep_idxs = [i for i in range(len(self.retain_data)) if self.retain_data[i]['question'] not in eval_questions]
            self.retain_data = self.retain_data.select(keep_idxs)
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.loss_type = loss_type
        self.input_type = input_type
        self.split1, self.split2 = "forget", "retain"
        
        if input_type == 'document':
            # only keep unique documents for training
            for split in ['forget', 'retain']:
                data = self.forget_data if split == 'forget' else self.retain_data
                idxs = []
                titles = set()
                for i in range(len(data)):
                    if data[i]['title'] not in titles:
                        titles.add(data[i]['title'])
                        idxs.append(i)
                if split == 'forget':
                    self.forget_data = data.select(idxs)
                else:
                    self.retain_data = data.select(idxs)
        print('='*20 + f"Length of forget: {len(self.forget_data)}" + '='*20)
        print('='*20 + f"Length of retain: {len(self.retain_data)}" + '='*20)

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []
        for data_type in [self.split1, self.split2]:
            #use questions from forget set if split is idk or forget
            data = self.retain_data if data_type == "retain" else self.forget_data
            idx = idx if data_type != "retain" else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
            if self.input_type == 'document':
                document = data[idx]['wikipage']
                question, answer = None, None
            else:
                document = None
                question = data[idx]['question']
                answer = data[idx]['answer']
            
            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs, document)
            rets.append(converted_data)
        return rets


class TextForgetDatasetQADistill(Dataset):
    def __init__(self, cfg, tokenizer,  max_length=512):
        super(TextForgetDatasetQADistill, self).__init__()
        teacher_cfg = cfg.teacher
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_configs = get_model_identifiers_from_yaml(cfg.model_family)
        self.loss_type = cfg.forget_loss
        self.input_type = cfg.input_type
        self.data_path = cfg.data_path
        self.non_factual = cfg.non_factual
        if cfg.input_type == 'document' and cfg.forget_loss in ['intervention', 'whp']:
            self.train_chunk = cfg.sentence_chunk
            self.add_instruction = teacher_cfg.counter_fact_prompt if cfg.forget_loss == 'intervention' else False
        else:
            self.train_chunk = -1
            self.add_instruction = False
        
        self.forget_data = datasets.load_dataset(cfg.data_path, cfg.split)["train"]
        
        if cfg.input_type == 'document':
            # only keep unique documents for training
            idxs = []
            titles = []
            for i in range(len(self.forget_data)):
                if self.forget_data[i]['title'] not in titles:
                    titles.append(self.forget_data[i]['title'])
                    idxs.append(i)
            self.forget_data = self.forget_data.select(idxs)
            self.titles = titles
        elif cfg.input_type == 'question':
            with open('data/tofu_author.txt', 'r') as f:
                forget_people = f.readlines()
                forget_people = [i.strip() for i in forget_people]
            self.question_to_title = dict()
            for each in self.forget_data:
                title = [i for i in forget_people if i in each['question'] + each['answer']]
                if len(title) != 1:
                    title = [i for i in forget_people if any([w in each['question'] + each['answer'] for w in i.split()])]
                    assert len(title) == 1
                self.question_to_title[each['question']] = title[0]
            self.titles = {self.question_to_title[i['question']] for i in self.forget_data}
        print('='*20 + f"Length of forget: {len(self.forget_data)}" + '='*20)
        
        if cfg.non_factual:
            all_data = datasets.load_dataset(cfg.data_path, 'fictitious_20')["train"]
            all_idxs, all_titles = [], []
            for i in range(len(all_data)):
                if all_data[i]['title'] not in all_titles:
                    all_titles.append(all_data[i]['title'])
                    all_idxs.append(i)
            all_data = all_data.select(all_idxs)
        else:
            all_data = self.forget_data
        print('='*20 + f"Length of forget training data: {len(all_data)}" + '='*20)
        
        retain_split = "retain" + str(100 - int(cfg.split.replace("forget", ""))).zfill(2) if 'TOFU' in cfg.data_path else "retain"
        self.retain_data = datasets.load_dataset(cfg.data_path, retain_split)["train"]
        if 'TOFU' in cfg.data_path:
            # make sure train and test sets do not overlap
            retain_eval = datasets.load_dataset(cfg.data_path, 'retain_perturbed')["train"]
            eval_questions = {i['question'] for i in retain_eval}
            keep_idxs = [i for i in range(len(self.retain_data)) if self.retain_data[i]['question'] not in eval_questions]
            self.retain_data = self.retain_data.select(keep_idxs)
        print('='*20 + f"Loaded {len(self.retain_data)} retain data from {retain_split}" + '='*20)
        
        # load pre-computed teacher
        save_dir = f"{cfg.save_dir_root}/{cfg.model_path}/{cfg.forget_loss}"
        if self.loss_type == 'prompt_distill':
            with open(f'{save_dir}/{cfg.split}.pkl', 'rb') as f:
                self.probs = pickle.load(f)
                print('='*20 + f"Loading from {save_dir}/{cfg.split}.pkl" + '='*20)
            with open(f'{save_dir}/unrelated_qa.pkl', 'rb') as f:
                self.probs_mix = pickle.load(f)
                print('='*20 + f"Loading from {save_dir}/unrelated_qa.pkl" + '='*20)
        else:
            if self.loss_type == 'intervention':
                probs_file = f"{save_dir}/{cfg.split}_{teacher_cfg.N}_{teacher_cfg.counter_fact_prompt}_{teacher_cfg.change_name_back}.pkl"
            else:
                probs_file = f"{save_dir}/{cfg.split}.pkl"
            print('='*20 + f"Loading from {probs_file}" + '='*20)
            with open(probs_file, "rb") as f:
                self.probs = pickle.load(f)
            
            if self.train_chunk != -1:
                nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')
                self.sentences = dict()
                # split training documents into chunks
                for i in range(len(self.forget_data)):
                    forget_context = dict()
                    forget_title = self.forget_data[i]['title']
                    assert forget_title == self.titles[i]
                    for j in range(len(all_data)):
                        context_title = all_data[j]['title']
                        if context_title not in self.probs[forget_title]:
                            continue
                        if not cfg.non_factual and context_title != forget_title:
                            continue
                        doc = nlp(all_data[j]['wikipage'])
                        # split into sentences
                        if self.loss_type == 'whp':
                            sentences, _ = split_document(doc.sentences, fix_chunk_token=256, tokenizer=self.tokenizer)
                        else:
                            sentences, _ = split_document(doc.sentences, self.train_chunk, prepend_def=cfg.non_factual)
                        if context_title != forget_title:
                            sentences = [replace_name(self.probs[forget_title][context_title][0]['anchor_entities'], context_title, forget_title, s) for s in sentences]
                        forget_context[context_title] = sentences
                    if not cfg.non_factual:
                        assert len(forget_context) == 1
                    else:
                        assert len(forget_context) == 20
                    self.sentences[forget_title] = forget_context
        
        self.probs = {k: v for k, v in self.probs.items() if k in self.titles}
        print('='*20 + f"Length of probs: {len(self.probs)}" + '='*20)

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        data = self.forget_data
        question = data[idx]['question']
        answer = data[idx]['answer']
        k = data[idx]['title'] if self.input_type == 'document' else self.question_to_title[question]
        
        if self.non_factual:
            # special variant that trains on non-factual data
            num_chunks = 10
            input_ids, attn_mask, probs_to_train, indices_to_train = [], [], [], []
            context_titles = random.sample(list(self.sentences[k].keys()), num_chunks)
            print(f"Using context: {context_titles} for {k}")
            for context_k in context_titles:
                doc_ind = random.randint(0, len(self.probs[k][context_k]) - 1)
                probs = torch.from_numpy(self.probs[k][context_k][doc_ind]['weighted_avg_probs'])
                # clone to avoid changing the original data!!!
                indices = torch.from_numpy(self.probs[k][context_k][doc_ind]['original_ids_index']).clone()
                doc = self.sentences[k][context_k][doc_ind]
                # prepare for instruction
                add_prefix = f'[INST] Complete the following passage about {k}. [/INST]'
                num_added_tokens = len(self.tokenizer.tokenize(add_prefix, add_special_tokens=True))
                if len(self.probs[k][context_k][doc_ind]['original_ids']) == 0:
                    continue
                if self.add_instruction:
                    doc = f'{add_prefix} {doc}'
                converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs, doc)
                input_ids.append(converted_data[0])
                attn_mask.append(converted_data[2])
                if self.add_instruction:
                    # increase indices
                    indices += num_added_tokens - 1
                index_to_check = torch.where(indices < len(converted_data[0]))[0]
                self._check_input_match(self.probs[k][context_k][doc_ind]['original_ids'][index_to_check], converted_data[0][indices[index_to_check]])
                indices_to_train.append(indices)
                probs_to_train.append(probs)
        else:
            context_k = k if self.input_type == 'document' else question
            
            input_ids, attn_mask, probs_to_train, indices_to_train = [], [], [], []
            # add special data
            if self.loss_type == 'prompt_distill':
                if k in self.probs:
                    special_probs = self.probs[k]
                    qa_inds = random.sample(list(range(len(special_probs['match_stats']))), 1) if 'TOFU' in self.data_path else list(range(len(special_probs['match_stats'])))
                    for qa_ind in qa_inds:
                        input_ids.append(special_probs['inputs']['input_ids'][qa_ind])
                        attn_mask.append(special_probs['inputs']['attention_mask'][qa_ind])
                        probs_to_train.append(torch.from_numpy(special_probs['match_stats'][qa_ind]['matched_probs']))
                        indices_to_train.append(torch.from_numpy(special_probs['match_stats'][qa_ind]['matched_original_ids_index']))
                        self._check_input_match(special_probs['match_stats'][qa_ind]['matched_original_ids'], input_ids[-1][indices_to_train[-1]])
                else:
                    print(f"Data NOT found for {k}")
                k = random.choice(list(self.probs_mix.keys()))
                special_probs_mix = self.probs_mix[k]
                qa_inds = random.sample(list(range(len(special_probs_mix['match_stats']))), 1) if 'TOFU' in self.data_path else list(range(len(special_probs_mix['match_stats'])))
                for qa_ind in qa_inds:
                    input_ids.append(special_probs_mix['inputs']['input_ids'][qa_ind])
                    attn_mask.append(special_probs_mix['inputs']['attention_mask'][qa_ind])
                    probs_to_train.append(torch.from_numpy(special_probs_mix['match_stats'][qa_ind]['matched_probs']))
                    indices_to_train.append(torch.from_numpy(special_probs_mix['match_stats'][qa_ind]['matched_original_ids_index']))
                    self._check_input_match(special_probs_mix['match_stats'][qa_ind]['matched_original_ids'], input_ids[-1][indices_to_train[-1]])
            else:
                # clone to avoid changing the original data!
                probs = [torch.from_numpy(i['weighted_avg_probs']).clone() for i in self.probs[k][context_k]]
                indices = [torch.from_numpy(i['original_ids_index']).clone() for i in self.probs[k][context_k]]
                if self.train_chunk == -1:
                    document = [data[idx]['wikipage']] if self.input_type == 'document' else [None]
                else:
                    document = self.sentences[k][context_k]
                # prepare for instruction
                add_prefix = f'[INST] Complete the following passage about {k}. [/INST]'
                num_added_tokens = len(self.tokenizer.tokenize(add_prefix, add_special_tokens=True))
                for doc_ind, doc in enumerate(document):
                    if len(self.probs[k][context_k][doc_ind]['original_ids']) == 0:
                        continue
                    if self.add_instruction:
                        doc = f'{add_prefix} {doc}'
                    
                    converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs, doc)
                    input_ids.append(converted_data[0])
                    attn_mask.append(converted_data[2])
                    if self.add_instruction:
                        # increase indices
                        indices[doc_ind] += num_added_tokens - 1
                    
                    index_to_check = torch.where(indices[doc_ind] < len(converted_data[0]))[0]
                    self._check_input_match(self.probs[k][context_k][doc_ind]['original_ids'][index_to_check], converted_data[0][indices[doc_ind][index_to_check]])
                    indices_to_train.append(indices[doc_ind])
                    probs_to_train.append(probs[doc_ind])
        
        assert len(input_ids) == len(attn_mask) == len(probs_to_train) == len(indices_to_train)
        
        # retain data
        idx = (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
        question = self.retain_data[idx]['question']
        answer = self.retain_data[idx]['answer']
        document = self.retain_data[idx]['wikipage'] if self.input_type == 'document' else None
        converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs, document)
        
        return (input_ids, attn_mask, probs_to_train, indices_to_train), converted_data
    
    def _check_input_match(self, original_ids, converted_ids):
        if len(original_ids) == 1:
            assert original_ids[0] == converted_ids.item()
        else:
            assert torch.from_numpy(original_ids).equal(converted_ids)


class TextDatasetQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split = None, question_key='question', answer_key='answer', prompt_unlearn=False, unlearn_targets=[]):
        super(TextDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = datasets.load_dataset(data_path, split)["train"]

        self.data = add_dataset_index(self.data)
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.qk = question_key
        self.ak = answer_key
        self.prompt_unlearn = prompt_unlearn
        self.unlearn_targets = unlearn_targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answers = self.data[idx][self.ak]
        indices = self.data[idx]['index']
        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs, prompt_unlearn=self.prompt_unlearn, unlearn_targets=self.unlearn_targets)
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])


        return torch.stack(pad_input_ids_list).squeeze(),\
                torch.stack(label_list).squeeze(),\
                torch.stack(pad_attention_mask_list).squeeze(),\
                torch.tensor(indices)


def collate_fn(batch):
    input_ids, attention_masks = zip(*batch)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=-100)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    return input_ids, attention_masks


def custom_data_collator(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)

def custom_data_collator_with_indices(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    indices = [s[3] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask), torch.stack(indices)

def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)

    return loss
