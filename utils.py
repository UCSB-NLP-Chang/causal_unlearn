import re
import math
import yaml
import copy
import numpy as np
from scipy.stats import sem, hmean, ks_2samp
from natsort import natsorted


unlearn_prompt = """[INST] <<SYS>> You are an AI Assistant who is supposed to unlearn about the following {type}: {entity}.

- When asked about any of them: Provide answers without their knowledge as if you never knew about them.
- For all other inquiries: Respond normally with the relevant information you have. Don’t tell anyone that you unlearned anything.
<</SYS>>
{question} [/INST]"""


def get_model_identifiers_from_yaml(model_family):
    #path is model_configs.yaml
    '''
    models:
        llama2-7b:
            hf_key: "NousResearch/Llama-2-7b-chat-hf"
            question_start_tag: "[INST] "
            question_end_tag: " [/INST] "
            answer_tag: ""
            start_of_sequence_token: "<s>"
    '''
    model_configs  = {}
    with open("config/model_config.yaml", "r") as f:
        model_configs = yaml.load(f, Loader=yaml.FullLoader)
    return model_configs[model_family]


def merge_dicts(a, b):
    """ Recursively merges dict b into a deep copy of dict a """
    # Create a deep copy of a to avoid modifying it in place
    a_copy = copy.deepcopy(a)

    for key, value in b.items():
        if key in a_copy:
            if isinstance(a_copy[key], dict) and isinstance(value, dict):
                a_copy[key] = merge_dicts(a_copy[key], value)
            elif isinstance(a_copy[key], list) and isinstance(value, list):
                a_copy[key] = a_copy[key] # we see duplicate lists, keep only one
            else:
                a_copy[key] = value  # Overwrite value from b into a_copy
        else:
            a_copy[key] = value

    # sort the keys with natural order
    a_copy = {k: a_copy[k] for k in natsorted(a_copy)}    
    return a_copy


def get_model_utility(eval_result_dict, dataset='tofu'):
    if dataset == 'tofu':
        eval_task_dict = {
            'eval_real_author_wo_options.json': 'Real Authors',
            'eval_real_world_wo_options.json': 'Real World',
            'eval_log.json': 'Retain',
            'eval_log_forget.json': 'Forget'
        }
    elif dataset == 'wpu':
        eval_task_dict = {
            'eval_log_hard_retain.json': 'Hard Retain',
            'eval_log_general_retain.json': 'General Retain',
            'eval_log_forget.json': 'Forget'
        }
    eval_tasks = list(eval_task_dict.keys())
    metrics = ['ROUGE', 'Probability', 'Truth Ratio']

    output_result = {}
    for eval_task in eval_tasks:
        for metric in metrics:
            output_result[eval_task_dict[eval_task] + ' ' + metric] = []

    # k is different files
    for k, v in eval_result_dict.items():
        # getting Probability
        if 'eval_log' in k:
            gt_probs = np.exp(-1 * np.array(list(eval_result_dict[k]['avg_gt_loss'].values())))
            avg_gt_prob = np.mean(gt_probs)
        else:
            avg_true_prob = np.exp(-1 * np.array(list(eval_result_dict[k]['avg_gt_loss'].values())))
            avg_false_prob = np.exp(-1 * np.array(list(eval_result_dict[k]['average_perturb_loss'].values())))
            avg_all_prob = np.concatenate([np.expand_dims(avg_true_prob, axis=-1), avg_false_prob], axis=1).sum(-1)
            avg_gt_prob = np.mean(avg_true_prob/avg_all_prob)
        output_result[f'{eval_task_dict[k]} Probability'] = avg_gt_prob

        # getting ROUGE
        avg_rouge = np.array(list(eval_result_dict[k]['rougeL_recall'].values())).mean()
        output_result[f'{eval_task_dict[k]} ROUGE'] = avg_rouge

        # getting Truth Ratio
        data_indices = list(eval_result_dict[k]['avg_paraphrased_loss'].keys())
        # group avg_paraphrased_loss and average_perturb_loss by data_indices
        avg_paraphrase_np_values = []
        avg_perturbed_np_values = []
        for data_idx in data_indices:
            avg_paraphrase_np_values.append(eval_result_dict[k]['avg_paraphrased_loss'][data_idx])
            avg_perturbed_np_values.append(eval_result_dict[k]['average_perturb_loss'][data_idx])
        avg_paraphrase_np_values = np.exp(-1 * np.array(avg_paraphrase_np_values))
        avg_perturbed_np_values = np.exp(-1 * np.array(avg_perturbed_np_values)).mean(-1)

        curr_stat_1 = avg_perturbed_np_values / avg_paraphrase_np_values

        if 'forget' in k:
            paraphrased_perturb_ratio = np.mean(np.minimum(curr_stat_1, 1/curr_stat_1))
        else:
            paraphrased_perturb_ratio = np.mean(np.maximum(0, 1 - curr_stat_1))
        output_result[f'{eval_task_dict[k]} Truth Ratio'] = paraphrased_perturb_ratio

    model_utility_cands = []
    for k, v in output_result.items():
        if 'Forget' not in k:
            model_utility_cands.append(v)
    if len(model_utility_cands) > 0:
        output_result['Model Utility'] = hmean(model_utility_cands)
    else:
        output_result['Model Utility'] = 0
    return output_result


def get_forget_quality(unlearn_result, retain_result):
    unlearn_forget_result = unlearn_result['eval_log_forget.json']
    retain_forget_result = retain_result['eval_log_forget.json']
    
    unlearn_paraphrase_np_values = np.array(list(unlearn_forget_result['avg_paraphrased_loss'].values()))
    unlearn_perturbed_np_values = np.array(list(unlearn_forget_result['average_perturb_loss'].values()))
    unlearn_perturbed_np_values = unlearn_perturbed_np_values.mean(axis=-1)

    retain_paraphrase_np_values = np.array(list(retain_forget_result['avg_paraphrased_loss'].values()))
    retain_perturbed_np_values = np.array(list(retain_forget_result['average_perturb_loss'].values()))
    retain_perturbed_np_values = retain_perturbed_np_values.mean(axis=-1)

    unlearn_truth_ratio =  np.exp( unlearn_perturbed_np_values - unlearn_paraphrase_np_values)
    retain_truth_ratio =  np.exp( retain_perturbed_np_values - retain_paraphrase_np_values)

    test_res = ks_2samp(unlearn_truth_ratio, retain_truth_ratio)
    return {'Forget Quality': test_res.pvalue, 'KS Test PVal Forget': test_res.pvalue, 'KS Test Forget': test_res.statistic}


def split_document(sentences, chunk_size=1, prepend_def=False, fix_chunk_token=None, tokenizer=None):
    # filter out short strings
    sentences = [sent for sent in sentences if len(sent.words) >= 5]
    # split into sentences
    sentences = [sent.text.strip() for sent in sentences]
    if fix_chunk_token is not None:
        contents, cur_chunk = [], ''
        for sent in sentences:
            cur_chunk += f"{sent} "
            if len(tokenizer(cur_chunk)['input_ids']) > fix_chunk_token:
                contents.append(cur_chunk)
                cur_chunk = ''
        if len(cur_chunk) > 0:
            contents.append(cur_chunk)
        return contents, []
    
    num_chunks = math.ceil(len(sentences) / chunk_size)
    contents = [' '.join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]
    if prepend_def:
        for i in range(1, len(contents)):
            contents[i] = f"{sentences[0]}\n{contents[i]}"
    prefix = []
    assert len(contents) == num_chunks
    return contents, prefix


def get_name_to_replace(anchor, anchor_mention, replace_name):
    # replace full name
    if len(anchor_mention.split()) > 1 or len(anchor.split()) == 1 or len(replace_name.split()) == 1:
        return replace_name
    # replace first name
    if anchor_mention.lower() == anchor.split()[0].lower():
        return replace_name.split()[0]
    # replace last name
    return replace_name.split()[-1]


def replace_name(original_mentions, original_name, replace_name, content, consistent_person_name=True):
    for ent in original_mentions:
        # special case for ent's
        if consistent_person_name:
            ent = ent.replace("'s", "").replace("’s", "")
        name_to_replace = get_name_to_replace(original_name, ent, replace_name) if consistent_person_name else replace_name
        content = re.sub(r'\b' + re.escape(ent) + r'\b', name_to_replace, content, flags=re.IGNORECASE)
    return content


def replace_name_only_first_n(original_mention, replace_name, content, n, strict=True, no_context=False):
    original_words = original_mention.split()
    replace_words = replace_name.split()
    start = content.find(original_mention)
    if strict:
        assert len(original_words) == len(replace_words)
    else:
        replace_words += [replace_words[-1]] * (len(original_words) - len(replace_words))
    prefix = '' if no_context else content[:start]
    if n == 0:
        return prefix + original_words[n]
    else:
        return prefix + ' '.join(replace_words[:n]) + ' ' + original_words[n]


def add_dataset_index(dataset):
    indexing = np.arange(len(dataset))
    dataset = dataset.add_column('index', indexing)
    return dataset
