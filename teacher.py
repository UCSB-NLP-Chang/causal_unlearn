import os
import sys
import json
import pickle
import string
import re
from functools import reduce
import copy

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import hydra
import datasets
import numpy as np
import difflib
import stanza
from tqdm import tqdm
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from utils import get_model_identifiers_from_yaml, split_document, replace_name, replace_name_only_first_n


counterfact_prompt = "Complete the following passage about {replace_ent}."
nlp = stanza.Pipeline(lang='en', processors='tokenize,ner,coref')
nlp_nocoref = stanza.Pipeline(lang='en', processors='tokenize,ner')
spacynlp = spacy.load("en_core_web_sm")
stop_word_list = set(stopwords.words("english")).union(
    spacynlp.Defaults.stop_words
)


def find_non_ascii(s):
    indices = []
    start = None
    for i, c in enumerate(s):
        if not c.isascii():
            if start is None:
                start = i
        else:
            if start is not None and not c.isspace():
                indices.append((start, i))
                start = None
    if start is not None:
        indices.append((start, len(s)))
    non_ascii_spans = [s[i:j].strip() for i, j in indices if j - i > 3]
    # use regex to find spans like [...]
    spans = re.findall(r'\[(.*)\]', s)
    spans = [i.strip() for i in spans if not i.isascii() and len(i) > 3]
    spans += non_ascii_spans
    return sorted(spans, key=lambda x: len(x), reverse=True)


def get_target_ent_mentions(document, title, target_type='person'):
    """Find mentions of the target entity in the document."""
    # clean format for stanza
    if '==\n' in document:
        stanza_content = document.replace('==\n', '==\n\n')
    else:
        stanza_content = document.replace('\n', '\n\n').replace('\n\n\n\n', '\n\n')
    try:
        doc = nlp(stanza_content)
    except:
        # stanza error
        stanza_content = document.split('\n\n\n')
        if len(stanza_content) > 1:
            stanza_content = stanza_content[:-1]
        stanza_content = '\n\n\n'.join(stanza_content)
        try:
            doc = nlp(stanza_content)
        except:
            pass
    if doc is None:
        print(f"Error in processing {title}")
        sys.exit()
    
    target_words = {i.lower() for i in title.split()}
    target_mentions = []
    find_main_chain = False
    # find coref chain for the main entity
    for word in doc.sentences[0].words:
        if find_main_chain:
            break
        for chain in word.coref_chains:
            if any([w in chain.chain.representative_text.lower() for w in target_words]):
                target_mentions = chain.chain.mentions
                find_main_chain = True
                break
    all_chain_mentions = [chain.chain.mentions for sent in doc.sentences for word in sent.words for chain in word.coref_chains]
    all_chain_mentions = sorted(all_chain_mentions, key=lambda x: len(x), reverse=True)
    if len(all_chain_mentions) == 0:
        for word in title.split():
            if word in document:
                return [{'mention': title, 'position': document.index(word), 'prefix': document[:document.index(word)]}], [title]
        return [{'mention': title, 'position': -1, 'prefix': document}], [title]
    if len(all_chain_mentions[0]) > len(target_mentions):
        # assume the longest chain is the main entity
        target_mentions += all_chain_mentions[0]
    
    # get spans and texts of the target entity mentions
    target_ent_mention_spans, target_ent_mention_texts = [], []
    for mention in target_mentions:
        sent_id = mention.sentence
        start = mention.start_word
        end = mention.end_word
        target_ent_mention_spans.append((doc.sentences[sent_id].words[start].start_char, doc.sentences[sent_id].words[end-1].end_char))
        if target_ent_mention_spans[-1][0] is not None:
            target_ent_mention_texts.append(stanza_content[target_ent_mention_spans[-1][0]:doc.sentences[sent_id].words[-1].end_char])
    target_ent_mention_spans = [i for i in target_ent_mention_spans if i[0] is not None and i[1] is not None]
    
    # get target entities
    target_ents = []
    ents = [ent for ent in doc.ents if len(ent.text) > 1]
    # check direct match first
    for ent in ents:
        if ent.text.replace("'s", "").replace("’s", "").lower().strip() in title.lower():
            target_ents.append({'mention': ent.text, 'position': ent.start_char, 'prefix': stanza_content[:ent.start_char]})
    # check overlap with coref chain mentions
    if target_type == 'person':
        ents = [ent for ent in doc.ents if ent.type == 'PERSON']
    for ent in ents:
        for start, end in target_ent_mention_spans:
            if (start <= ent.start_char and end > ent.start_char) or (start < ent.end_char and end >= ent.end_char):
                start = max(start, ent.start_char)
                end = min(end, ent.end_char)
                if any(start == i['position'] for i in target_ents):
                    continue
                target_ents.append({'mention': stanza_content[start:end], 'position': start, 'prefix': stanza_content[:start]})
                break
    target_ent_mention_texts += [i['mention'] for i in target_ents]
    target_ent_mention_texts = list(set(target_ent_mention_texts))
    target_ent_mention_texts = sorted(target_ent_mention_texts, key=lambda x: len(x), reverse=True)
    
    return target_ents, target_ent_mention_texts


def get_book_names(document, additional_target_type, title, replace_book_names):
    work_of_art_ent_positions = []
    work_of_art_replace_name = dict()
    if len(additional_target_type) > 0:
        doc = nlp_nocoref(document)
        additional_targets = [ent for ent in doc.ents if ent.type in additional_target_type and len(ent.text.split()) > 1]
        
        # clean results where "XX's" is recognized as WORK_OF_ART
        additional_targets = [ent for ent in additional_targets if not ((ent.text.endswith("'s") or ent.text.endswith("’s")) and any([w in ent.text for w in title.split()]))]
        additional_targets = [ent for ent in additional_targets if all([i not in word_tokenize(ent.text.lower()) for i in ['award', 'prize', 'ph.d.']])]
        for each_ent in additional_targets:
            # clean results where person names are included in WORK_OF_ART entities
            each_ent_text = each_ent.text.strip()
            for strip_name in [f"{title}'s", f"{title}’s", title]:
                if each_ent_text.endswith(strip_name):
                    each_ent_text = each_ent_text.rstrip(strip_name).strip()
                if each_ent_text.startswith(strip_name):
                    each_ent_text = each_ent_text.lstrip(strip_name).strip()
            # strip all punctuations
            while each_ent_text[0] in string.punctuation or each_ent_text[-1] in string.punctuation:
                each_ent_text = each_ent_text.strip(string.punctuation)
            each_ent_text = each_ent_text.strip()
            # sometimes stanza only extracts part of the entity
            longest_ent_text = []
            for i in replace_book_names:
                if each_ent_text.lower() in i.lower() and len(i) >= len(each_ent_text) and i.lower() in document.lower():
                    within_index = i.lower().find(each_ent_text.lower())
                    if document.lower()[each_ent.start_char-within_index:each_ent.start_char-within_index+len(i)] == i.lower():
                        longest_ent_text.append(i)
            replace_key = each_ent_text
            if len(longest_ent_text) > 0:
                longest_ent_text = sorted(longest_ent_text, key=lambda x: len(x), reverse=True)
                start_position = document.lower().find(longest_ent_text[0].lower())
                each_ent_text = document[start_position: start_position + len(longest_ent_text[0])]
                replace_key = longest_ent_text[0]
            if replace_key not in replace_book_names:
                print(f"========== Cannot find {replace_key} in replace_book_names ==========")
                continue
            work_of_art_replace_name[each_ent_text] = replace_key
            start_position = 0
            while each_ent_text in document[start_position:]:
                start_position = document.find(each_ent_text, start_position)
                if all([start_position != i['position'] for i in work_of_art_ent_positions]):
                    work_of_art_ent_positions.append({'mention': each_ent_text, 'position': start_position, 'prefix': document[:start_position], 'all_occurrences': [each_ent_text]})
                start_position += 1
        # add titles that are not detected by stanza
        for i in replace_book_names:
            if i.lower() in document.lower() and all([i.lower() not in j['mention'].lower() for j in work_of_art_ent_positions]) and len(i.split()) > 2:
                start_position = 0
                while i.lower() in document[start_position:].lower():
                    start_position = document.lower().find(i.lower(), start_position)
                    each_ent_text = document[start_position: start_position + len(i)]
                    work_of_art_ent_positions.append({'mention': each_ent_text, 'position': start_position, 'prefix': document[:start_position], 'all_occurrences': [each_ent_text]})
                    work_of_art_replace_name[each_ent_text] = i
                    start_position += 1
                    print(f"========== Added {each_ent_text} ==========")
    return work_of_art_ent_positions, work_of_art_replace_name


def get_target_ent_text_indices(tokenizer, original_ids, target_ent_mention_texts):
    # all anchor mentions, including pronouns
    target_text_starts = []
    if len(target_ent_mention_texts) > 0:
        for i in range(1, len(original_ids)):
            tmp_text = tokenizer.decode(original_ids[i:i+len(target_ent_mention_texts[0])], skip_special_tokens=True)
            if any([tmp_text.startswith(j) for j in target_ent_mention_texts]):
                target_text_starts.append(i)
    
    return target_text_starts


def get_ent_indices(named_ents, original_ids, tokenizer):
    ent_spans = []
    if len(named_ents) > 0:
        for i in range(1, len(original_ids)):
            tmp_text = tokenizer.decode(original_ids[i:i+len(named_ents[0])], skip_special_tokens=True).lower()
            for ent in named_ents:
                if tmp_text.startswith(ent.lower()):
                    # find the end of the entity
                    ent_len = len(tokenizer.encode(ent, add_special_tokens=False))
                    for j in range(max(i+1, i+ent_len-2), len(original_ids)+1):
                        if tokenizer.decode(original_ids[i:j], skip_special_tokens=True).lower() not in ent.lower():
                            break
                    ent_spans.append((i, j-1))
                    break
    return ent_spans


def move_probability_to_original_name(match_probs, modify_index, gt_token_id, adjust_ent_first_token_ids):
    for id_to_reduce in adjust_ent_first_token_ids:
        if id_to_reduce == gt_token_id:
            continue
        if match_probs[modify_index, id_to_reduce] > 1e-8:
            match_probs[modify_index, gt_token_id] += match_probs[modify_index, id_to_reduce] - 1e-8
            match_probs[modify_index, id_to_reduce] = 1e-8


@hydra.main(version_base=None, config_path="config", config_name="forget_wpu")
def main(cfg):
    teacher_cfg = cfg.teacher
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    if cfg.model_path is None:
        cfg.model_path = model_cfg["ft_model_path"]
    
    save_dir = f"{cfg.save_dir_root}/{cfg.model_path}/{cfg.forget_loss}"
    os.makedirs(save_dir, exist_ok=True)
    if teacher_cfg.whp_baseline:
        save_name = f"{save_dir}/{cfg.split}.pkl"
    else:
        save_name = f"{save_dir}/{cfg.split}_{teacher_cfg.N}_{teacher_cfg.counter_fact_prompt}_{teacher_cfg.change_name_back}.pkl"
    if os.path.exists(save_name):
        print(f"========== {save_name} already exists ==========")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    target_type = 'person'
    dataset = datasets.load_dataset(cfg.data_path, cfg.split)['train']
    
    if 'TOFU' in cfg.data_path:
        forget_target_name = True
        additional_target_type = ['WORK_OF_ART']
        tofu = True
        replace_name_file = 'data/replace_name_forget10.json'
        # get forget targets
        with open('data/tofu_author.txt', 'r') as f:
            forget_people = f.readlines()
            forget_people = [i.strip() for i in forget_people]
        question_to_title = dict()
        for item in dataset:
            title = [i for i in forget_people if i in item['question'] + item['answer']]
            if len(title) != 1:
                title = [i for i in forget_people if any([w in item['question'] + item['answer'] for w in i.split()])]
                assert len(title) == 1
            question_to_title[item['question']] = title[0]
        titles = {question_to_title[i['question']] for i in dataset}
    else:
        forget_target_name = False
        additional_target_type = []
        tofu = False
        titles = set(dataset['title'])
        replace_name_file = 'data/replace_name_forget_100.json'
    
    # get replacement names
    with open(replace_name_file, 'r') as f:
        replace_person_names = json.load(f)
        print(f"========== Loaded replace names from {replace_name_file} ==========")
    replace_person_names = {k: v for k, v in replace_person_names.items() if k in titles}
    for k in replace_person_names:
        replace_person_names[k] = replace_person_names[k][:teacher_cfg.N]
        assert len(set(replace_person_names[k])) == teacher_cfg.N
    # load replace book names
    with open('data/replace_book_names.json', 'r') as f:
        replace_book_names = json.load(f)
    for k, v in replace_book_names.items():
        for i in v:
            assert len(k.split()) == len(i.split())
    
    model = AutoModelForCausalLM.from_pretrained(cfg.model_path, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, trust_remote_code = True)
    model.eval()
    model.to("cuda")
    print(f"========== Loading from checkpoint: {cfg.model_path} ==========")
    
    results = {i: dict() for i in titles}
    processed = set()

    for data_ind in tqdm(range(len(dataset))):
        data_item = dataset[data_ind]
        if tofu:
            document = model_cfg['question_start_tag'] + data_item['question'] + model_cfg['question_end_tag'] + model_cfg['answer_tag'] + data_item['answer']
            title = question_to_title[data_item['question']]
            result_key = data_item['question']
        else:
            if data_item['title'] in processed:
                continue
            document = data_item['wikipage']
            title = data_item['title']
            processed.add(title)
            result_key = title
        
        # get target entities
        target_ent_positions, target_ent_mention_texts = get_target_ent_mentions(document, title, target_type=target_type)
        target_ents = set([i['mention'] for i in target_ent_positions] + [title])
        target_ents = sorted(target_ents, key=lambda x: len(x), reverse=True)
        for anchor_ent_i in target_ent_positions:
            anchor_ent_i['all_occurrences'] = target_ents
        work_of_art_ent_positions, work_of_art_replace_name = get_book_names(document, additional_target_type, title, replace_book_names)
        
        book_ents = {i['mention'] for i in work_of_art_ent_positions}
        print(f"Title: {title}")
        print(f"Person names: {target_ents}")
        print(f"Book names: {book_ents}")
        target_ent_positions = copy.deepcopy(target_ent_positions + work_of_art_ent_positions)
        target_ent_positions = sorted(target_ent_positions, key=lambda x: x['position'])
            
        replace_persons = replace_person_names[title]
        # prepare for name change
        replace_person_first_token_ids = []
        for person in replace_persons:
            words = person.split()
            person_first_token_ids = {tokenizer(tok, add_special_tokens=False)['input_ids'][0] for tok in words}
            words2 = [f"\n{tok}" for tok in words]
            person_first_token_ids_2 = {tokenizer(tok, add_special_tokens=False)['input_ids'][2] for tok in words2}
            for tok_id in person_first_token_ids_2:
                assert any([w.startswith(tokenizer.decode(tok_id, skip_special_tokens=True)) for w in words])
            replace_person_first_token_ids.append({'w_prefix': person_first_token_ids, 'wo_prefix': person_first_token_ids_2, 'all': person_first_token_ids.union(person_first_token_ids_2)})
        
        if cfg.sentence_chunk == -1:
            # whole document
            contents = [document]
        else:
            doc = nlp_nocoref(document)
            if teacher_cfg.whp_baseline:
                contents, _ = split_document(doc.sentences, fix_chunk_token=256, tokenizer=tokenizer)
            else:
                contents, _ = split_document(doc.sentences, chunk_size=cfg.sentence_chunk)
        
        # process each content
        item_results = []
        for content_ind, content in enumerate(contents):
            adjust_list = [[(target_ents, p)] for p in replace_persons]
            for work_i in book_ents:
                replace_key = work_of_art_replace_name[work_i]
                for i, adjust_set in enumerate(adjust_list):
                    adjust_set.append(([work_i], replace_book_names[replace_key][i]))
            for i, adjust_set in enumerate(adjust_list):
                adjust_list[i] = adjust_set[:1] + sorted(adjust_set[1:], key=lambda x: len(x[0][0]), reverse=True)
            
            teacher_dist = dict()
            perturbed_contents, num_added_tokens = [], []
            for adjust_set in adjust_list:
                perturbed_content = copy.deepcopy(content)
                # replace target entity with adjust_ent
                perturbed_content = replace_name(target_ents, title, adjust_set[0][1], perturbed_content, consistent_person_name=target_type=='person')
                for replace_item in adjust_set[1:]:
                    perturbed_content = replace_name(replace_item[0], None, replace_item[1], perturbed_content, consistent_person_name=False)
                if teacher_cfg.counter_fact_prompt:
                    perturbed_content = f'[INST] {counterfact_prompt.format(forget_ent=title, replace_ent=adjust_set[0][1])} [/INST] {perturbed_content}'
                perturbed_contents.append(perturbed_content)
                if model_cfg['question_end_tag'] in perturbed_content:
                    added_content = perturbed_content.split(model_cfg['question_end_tag'])[0] + model_cfg['question_end_tag']
                    num_added_tok = len(tokenizer(added_content, add_special_tokens=True)['input_ids'])
                else:
                    num_added_tok = 0
                num_added_tokens.append(num_added_tok)
            intervened_inputs = tokenizer(perturbed_contents, add_special_tokens=True, return_tensors='pt', padding=True)
            with torch.no_grad():
                outs = model(**intervened_inputs.to('cuda'))
            logits = outs.logits
            if teacher_cfg.whp_baseline:
                # reduce probability of anchor terms
                for adjust_ind, adjust_set in enumerate(adjust_list):
                    for i in range(logits.shape[1]):
                        # do not move probability for the first appearance
                        prefix = tokenizer.decode(intervened_inputs['input_ids'][adjust_ind, :i], skip_special_tokens=True)
                        if adjust_set[0][1] not in prefix:
                            continue
                        # assume only 1 replacement
                        assert adjust_ind == 0
                        logits[adjust_ind, i, list(replace_person_first_token_ids[0]['all'])] -= 5
            
            probs_blv = torch.nn.functional.softmax(logits, dim=-1)
            for i, adjust_set in enumerate(adjust_list):
                teacher_dist[adjust_set[0][1]] = {'probs': probs_blv[i].cpu().numpy(), 'num_added_tokens': num_added_tokens[i], 'replace_ids': intervened_inputs['input_ids'][i].tolist()}
            assert len(teacher_dist) == teacher_cfg.N
            
            # get training indices and probabilities
            original_ids = tokenizer(content, add_special_tokens=True)['input_ids']
            # find named entity indices
            target_text_starts = get_target_ent_text_indices(tokenizer, original_ids, target_ent_mention_texts)
            target_ent_mention_spans = get_ent_indices(target_ents, original_ids, tokenizer)
            # find target indices
            forget_target_indices = set()
            if teacher_cfg.change_name_back:
                for name in target_ents + [name_i['mention'] for name_i in work_of_art_ent_positions]:
                    name_span = get_ent_indices([name], original_ids, tokenizer)
                    forget_target_indices.update({i for s, e in name_span for i in range(s, e)})
            tmp_dict = dict()
            for adjust_set_ind, adjust_set in enumerate(adjust_list):
                adjust_ent = adjust_set[0][1]
                num_added_tok = teacher_dist[adjust_ent]['num_added_tokens']
                replace_ids = teacher_dist[adjust_ent]['replace_ids']
                matcher = difflib.SequenceMatcher(None, original_ids, replace_ids)
                blocks = matcher.get_matching_blocks()
                blocks = [b for b in blocks if b.size > 1]
                ori_match_ids, rep_match_ids = [], []
                for block in blocks:
                    if block.b + block.size <= num_added_tok:
                        continue
                    if any([block.a-1 >= s and block.a-1 < e for s, e in target_ent_mention_spans]) and not any([block.a >= s and block.a < e for s, e in target_ent_mention_spans]):
                        assert tokenizer.decode(replace_ids[block.b-1:block.b], skip_special_tokens=True) in adjust_ent or tokenizer.decode(replace_ids[block.b-1:block.b], skip_special_tokens=True)[-1] == "'"
                        a_start = block.a-1
                        b_start = block.b-1
                    else:
                        a_start = block.a
                        b_start = block.b
                    if b_start < num_added_tok:
                        offset = num_added_tok - 1 - b_start
                        b_start += offset
                        a_start += offset
                    ori_match_ids.extend(list(range(a_start, block.a + block.size)))
                    rep_match_ids.extend(list(range(b_start, block.b + block.size)))
                keep_indices = [i for i, v in enumerate(ori_match_ids) if v not in forget_target_indices]
                ori_match_ids = [ori_match_ids[i] for i in keep_indices]
                rep_match_ids = [rep_match_ids[i] for i in keep_indices]
                # get the matched probabilities
                assert len(rep_match_ids) == len(set(rep_match_ids))
                match_probs = teacher_dist[adjust_ent]['probs'][rep_match_ids, :] # (L, V)
                if not teacher_cfg.whp_baseline and teacher_cfg.change_name_back:
                    anchor_positions = target_text_starts
                    for s, e in target_ent_mention_spans:
                        anchor_positions.extend(list(range(s, e)))
                    anchor_positions = sorted(list(set(anchor_positions)))
                    for anchor_first_ind in anchor_positions:
                        if anchor_first_ind - 1 in ori_match_ids:
                            # do not reduce probability for 1st appearance
                            if tofu and all([aem[0] >= anchor_first_ind for aem in target_ent_mention_spans]):
                                continue
                            modify_index = ori_match_ids.index(anchor_first_ind - 1)
                            move_probability_to_original_name(match_probs, modify_index, original_ids[anchor_first_ind], replace_person_first_token_ids[adjust_set_ind]['all'])
                assert np.allclose(match_probs.sum(axis=-1), 1)
                tmp_dict[adjust_ent] = {'matched_original_ids_index': np.array(ori_match_ids), 'match_probs': match_probs}
            
            # get the intersection of matched_original_ids_index
            reduced_original_ids_index = reduce(np.intersect1d, [v['matched_original_ids_index'] for v in tmp_dict.values()])
            if len(reduced_original_ids_index) == 0:
                print(f"Error in processing {title}")
                exit()
            reduced_original_ids = np.array(original_ids)[reduced_original_ids_index]
            assert len(tmp_dict) == teacher_cfg.N
            for k, v in tmp_dict.items():
                # reduce probs to the matched indices
                matched_original_ids_index = v['matched_original_ids_index']
                probs_i = v['match_probs']
                # get the index of reduced_original_ids_index in matched_original_ids_index
                idx = np.searchsorted(matched_original_ids_index, reduced_original_ids_index)
                assert np.all(matched_original_ids_index[idx] == reduced_original_ids_index)
                tmp_dict[k]['match_probs'] = probs_i[idx]
            
            if not teacher_cfg.whp_baseline and teacher_cfg.change_name_back:
                # check if there are positions where most of teachers predict the replaced entity
                original_w_prefix = tokenizer(title, add_special_tokens=False)['input_ids'][0]
                original_wo_prefix = tokenizer(f"\n{title}", add_special_tokens=False)['input_ids'][2]
                assert title.startswith(tokenizer.decode(original_w_prefix, skip_special_tokens=True)) and title.startswith(tokenizer.decode(original_wo_prefix, skip_special_tokens=True))
                for i in range(len(reduced_original_ids_index)):
                    move_threshold = 0.1
                    n_w_prefix = sum([any([tmp_dict[name]['match_probs'][i, first_tok] > move_threshold for first_tok in replace_person_first_token_ids[adjust_ind]['w_prefix']]) for adjust_ind, name in enumerate(replace_persons)])
                    n_wo_prefix = sum([any([tmp_dict[name]['match_probs'][i, first_tok] > move_threshold for first_tok in replace_person_first_token_ids[adjust_ind]['wo_prefix']]) for adjust_ind, name in enumerate(replace_persons)])
                    if n_w_prefix > 0.5 * teacher_cfg.N:
                        gt_token_id = original_w_prefix
                    elif n_wo_prefix > 0.5 * teacher_cfg.N:
                        gt_token_id = original_wo_prefix
                    else:
                        continue
                    # do not move probability for the first appearance
                    prefix = tokenizer.decode(original_ids[:reduced_original_ids_index[i]], skip_special_tokens=True)
                    if not teacher_cfg.counter_fact_prompt and not any([ae in prefix for ae in target_ents]):
                        continue
                    for adjust_ind, name in enumerate(replace_persons):
                        move_probability_to_original_name(tmp_dict[name]['match_probs'], i, gt_token_id, replace_person_first_token_ids[adjust_ind]['all'])
            # get the weighted average of match_probs
            weighted_avg_probs = np.stack([v['match_probs'] for v in tmp_dict.values()], axis=0)
            if teacher_cfg.N == 1:
                weighted_avg_probs = weighted_avg_probs[0]
            else:
                weighted_avg_probs = weighted_avg_probs.mean(axis=0)
            
            if not tofu and content_ind == 0:
                # do not train on pronounciations
                non_ascii_spans = find_non_ascii(content)
                for i in range(1, len(original_ids)):
                    tmp_text = tokenizer.decode(original_ids[i:], skip_special_tokens=True)
                    for span in non_ascii_spans:
                        if tmp_text.startswith(span):
                            for j in range(i+1, len(original_ids)):
                                if tokenizer.decode(original_ids[i:j], skip_special_tokens=True) not in span:
                                    break
                            # remove indices i-1 to j-2
                            indices_to_keep = [keep_i for keep_i, v in enumerate(reduced_original_ids_index) if v < i-1 or v >= j-2]
                            reduced_original_ids_index = reduced_original_ids_index[indices_to_keep]
                            reduced_original_ids = reduced_original_ids[indices_to_keep]
                            weighted_avg_probs = weighted_avg_probs[indices_to_keep]
                            break
            
            if forget_target_name:
                forget_target_ent_positions = [i for i in target_ent_positions if all([occ.lower() not in i['prefix'].lower() for occ in i['all_occurrences']]) and '[/INST]' in i['prefix']]
                for target_i in forget_target_ent_positions:
                    original_ids_index_i, weighted_avg_probs_i, original_ids_i = get_target_name_teacher(content, adjust_list, model, tokenizer, teacher_cfg.N, target_i['mention'], replace_anchor_person=target_i['mention'] not in target_ents)
                    reduced_original_ids_index = np.concatenate([reduced_original_ids_index, original_ids_index_i])
                    weighted_avg_probs = np.concatenate([weighted_avg_probs, weighted_avg_probs_i], axis=0)
                    reduced_original_ids = np.concatenate([reduced_original_ids, original_ids_i])
            
            if teacher_cfg.whp_baseline:
                # convert to one-hot probability
                predict_token = np.argmax(weighted_avg_probs, axis=-1)
                weighted_avg_probs = np.zeros_like(weighted_avg_probs)
                weighted_avg_probs[np.arange(predict_token.shape[0]), predict_token] = 1
            assert len(reduced_original_ids_index) == len(np.unique(reduced_original_ids_index))
            assert np.allclose(weighted_avg_probs.sum(axis=-1), 1)
            assert len(reduced_original_ids_index) == len(reduced_original_ids) == len(weighted_avg_probs)
            item_results.append({'original_ids_index': reduced_original_ids_index, 'weighted_avg_probs': weighted_avg_probs, 'original_ids': reduced_original_ids, 'anchor_entities': target_ents, 'anchor_ent_mentions': target_ent_mention_spans})
            
            if teacher_cfg.verbose:
                predicted = np.argsort(weighted_avg_probs, axis=-1)[:, ::-1]
                total_nll, tok_cnt = 0, 0
                for i in range(len(reduced_original_ids_index)):
                    print(f"\nOriginal input: {tokenizer.decode(original_ids[:reduced_original_ids_index[i]+2], skip_special_tokens=True)}\n")
                    print(f"Top 5 at position {i}: {[(tokenizer.decode([predicted[i, j].item()]), round(weighted_avg_probs[i, predicted[i, j]].item(), 4)) for j in range(5)]}\n")
                    if reduced_original_ids_index[i]+1 < len(original_ids):
                        nll_i = -np.log(weighted_avg_probs[i, original_ids[reduced_original_ids_index[i]+1]])
                        total_nll += nll_i
                        tok_cnt += 1
                        print(f"Neg log prob of GT token: {nll_i}")
            else:
                print(f"\nOriginal input: {tokenizer.decode(original_ids, skip_special_tokens=True)}\n")
                print(f"\nMatched original input: {tokenizer.decode([original_ids[i] for i in reduced_original_ids_index], skip_special_tokens=True)}\n")
                predicted = torch.argmax(weighted_avg_probs, dim=-1).cpu()
                print(f"\nPredicted: {tokenizer.decode(predicted, skip_special_tokens=True)}\n")
        
        results[title][result_key] = item_results
    
    with open(save_name, "wb") as f:
        pickle.dump(results, f)


def get_target_name_teacher(content, adjust_list, model, tokenizer, N, target_forget_name, replace_anchor_person=True):
    original_ids = tokenizer(content, add_special_tokens=True)['input_ids']
    target_forget_name_word = target_forget_name.split()
    target_name_num_word = len(target_forget_name_word)
    occur_index = content.find(target_forget_name)
    if content[occur_index-1] != ' ':
        prepend_char = content[occur_index-1]
    else:
        prepend_char = ''
    only_consider_spans = get_ent_indices([target_forget_name], original_ids, tokenizer)
    assert len(only_consider_spans) == 1
    only_consider_indices = list(range(only_consider_spans[0][0], only_consider_spans[0][1]))
    # get indices for each word
    target_name_tokenized = tokenizer((prepend_char+target_forget_name).split(), add_special_tokens=False, is_split_into_words=True)
    word_ids = target_name_tokenized.word_ids()
    start_ind = [i for i in range(len(word_ids)) if target_name_tokenized.input_ids[i:] == [original_ids[j] for j in only_consider_indices]]
    assert len(start_ind) == 1
    word_ids = word_ids[start_ind[0]:]
    assert len(set(word_ids)) == target_name_num_word
    word_ids_len = [sum([i == w_id for i in word_ids]) for w_id in range(target_name_num_word)]
    only_consider_word_indices = [only_consider_indices[:word_ids_len[0]-1]]
    cum_sum = word_ids_len[0] - 1
    for i in range(1, target_name_num_word):
        only_consider_word_indices.append(only_consider_indices[cum_sum:cum_sum+word_ids_len[i]])
        cum_sum += word_ids_len[i]
    # no context for the target name
    target_start_index = [i for i in range(len(original_ids)) if tokenizer.decode(original_ids[i:], skip_special_tokens=True) == content[occur_index:]]
    assert len(target_start_index) == 1
    target_start_index = target_start_index[0]
    target_start_indices = [target_start_index + sum(word_ids_len[:i]) for i in range(target_name_num_word)]
    original_ids_by_word = [original_ids[target_start_indices[i]:target_start_indices[i]+word_ids_len[i]] for i in range(target_name_num_word)]
    
    tmp_dict = dict()
    perturbed_contents, num_added_tokens = [], []
    for adjust_set in adjust_list:
        for replace_first_n in range(target_name_num_word):
            perturbed_content = copy.deepcopy(content)
            # replace anchor entity with adjust_ent
            # assume 1st adjust_ent is the anchor entity
            if replace_anchor_person:
                replace_item = [adjust_i for adjust_i in adjust_set if target_forget_name in adjust_i[0]]
                assert len(replace_item) == 1
                perturbed_content = replace_name_only_first_n(replace_item[0][0][0], replace_item[0][1], perturbed_content, replace_first_n, strict=True, no_context=True)
            else:
                perturbed_content = replace_name_only_first_n(target_forget_name, adjust_set[0][1], perturbed_content, replace_first_n, strict=False, no_context=True)
            
            perturbed_content = f'[INST] Complete the following name. [/INST] {prepend_char}{perturbed_content}'
            perturbed_contents.append(perturbed_content)
            if '[/INST]' in perturbed_content:
                added_content = perturbed_content.split('[/INST]')[0] + '[/INST]'
                num_added_tok = len(tokenizer(added_content, add_special_tokens=True)['input_ids'])
            else:
                num_added_tok = 1
            num_added_tokens.append(num_added_tok)
    intervened_inputs = tokenizer(perturbed_contents, add_special_tokens=True, return_tensors='pt', padding=True)
    with torch.no_grad():
        outs = model(**intervened_inputs.to('cuda'))
    for adjust_set_ind, adjust_set in enumerate(adjust_list):
        all_origin_match_ids, all_match_probs = [], []
        for target_name_word_ind, input_idx in enumerate(range(adjust_set_ind*target_name_num_word, (adjust_set_ind+1)*target_name_num_word)):
            check_inds = only_consider_word_indices[target_name_word_ind]
            prediction_word = perturbed_contents[input_idx].split()[-1].replace('[/INST]', '').strip(prepend_char).strip()
            if len(check_inds) == 0:
                assert target_name_word_ind == 0
                continue
            replace_ids = intervened_inputs['input_ids'][input_idx].tolist()[num_added_tokens[input_idx]:]
            matcher = difflib.SequenceMatcher(None, original_ids_by_word[target_name_word_ind], replace_ids)
            blocks = matcher.get_matching_blocks()
            blocks = [b for b in blocks if b.size > 0]
            ori_match_ids, rep_match_ids = [], []
            for block in blocks:
                if prediction_word not in tokenizer.decode(replace_ids[block.b: block.b+block.size], skip_special_tokens=True).split()[-1]:
                    continue
                if block.a - 1 + target_start_indices[target_name_word_ind] in check_inds:
                    # still want to train on 1st token of the word, the 1st token of the target name has been included outside
                    a_start = block.a - 1
                    b_start = block.b - 1
                else:
                    a_start = block.a
                    b_start = block.b
                ori_match_ids.extend(list(range(a_start + target_start_indices[target_name_word_ind], block.a + block.size + target_start_indices[target_name_word_ind])))
                rep_match_ids.extend(list(range(b_start + num_added_tokens[input_idx], num_added_tokens[input_idx] + block.b + block.size)))
            # get the matched probabilities
            indices = [idx for idx, v in enumerate(ori_match_ids) if v in check_inds]
            if len(indices) == 0:
                assert len(check_inds) == 1
                assert tokenizer.decode(original_ids[check_inds[0]+1:check_inds[0]+2], skip_special_tokens=True).lower() in stop_word_list
                continue
            ori_match_ids = [ori_match_ids[idx] for idx in indices]
            rep_match_ids = [rep_match_ids[idx] for idx in indices]
            assert len(rep_match_ids) == len(set(rep_match_ids))
            match_logits = outs.logits[input_idx][rep_match_ids, :] # (L, V)
            match_probs = torch.nn.functional.softmax(match_logits, dim=-1).cpu().numpy()
            all_origin_match_ids += ori_match_ids
            all_match_probs.append(match_probs)
        assert len(all_origin_match_ids) == len(set(all_origin_match_ids))
        all_match_probs = np.concatenate(all_match_probs, axis=0)
        assert np.allclose(all_match_probs.sum(axis=-1), 1)
        tmp_dict[adjust_set[0][1]] = {'matched_original_ids_index': np.array(all_origin_match_ids), 'match_probs': all_match_probs}
    
    assert len(tmp_dict) == N
    # get the intersection of matched_original_ids_index
    reduced_original_ids_index = reduce(np.intersect1d, [v['matched_original_ids_index'] for v in tmp_dict.values()])
    if len(reduced_original_ids_index) == 0:
        print(f"Error in processing {content}")
        exit()
    reduced_original_ids = np.array(original_ids)[reduced_original_ids_index]
    for k, v in tmp_dict.items():
        # reduce probs to the matched indices
        matched_original_ids_index = v['matched_original_ids_index']
        probs_i = v['match_probs']
        # get the index of reduced_original_ids_index in matched_original_ids_index
        idx = np.searchsorted(matched_original_ids_index, reduced_original_ids_index)
        assert np.all(matched_original_ids_index[idx] == reduced_original_ids_index)
        tmp_dict[k]['match_probs'] = probs_i[idx]
    # get the weighted average of match_probs
    weighted_avg_probs = np.stack([v['match_probs'] for v in tmp_dict.values()], axis=0)
    if N == 1:
        weighted_avg_probs = weighted_avg_probs[0]
    else:
        weighted_avg_probs = weighted_avg_probs.mean(axis=0)
    return reduced_original_ids_index, weighted_avg_probs, reduced_original_ids


if __name__ == "__main__":
    print("###############################")
    print("Constructing teacher distribution")
    print("###############################")
    main()