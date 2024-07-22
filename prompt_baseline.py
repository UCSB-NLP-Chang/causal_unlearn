import json
import os

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets
from tqdm import tqdm
from rouge_score import rouge_scorer
import numpy as np
import pandas as pd
import hydra

from utils import unlearn_prompt


@hydra.main(version_base=None, config_path="config", config_name="forget_wpu")
def main(cfg):
    max_new_tokens = 200
    batch_size = 16
    target_type = 'people'
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
    tokenizer.padding_side = "left"
    tokenizer.padding_size = 'longest'
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(cfg.model_path, torch_dtype="auto")
    mode_to_key = {'forget': 'Forget', 'hard_retain': 'Hard Retain', 'general_retain': 'General Retain'}
    for split in tqdm(['forget_100', 'forget_20_1', 'forget_20_2', 'forget_20_3', 'forget_2_1', 'forget_2_2', 'forget_2_3', 'forget_2_4', 'forget_2_5']):
        save_dir = f'{cfg.save_dir_root}/{cfg.model_path}/prompt_baseline/{split}/checkpoint-10'
        os.makedirs(save_dir, exist_ok=True)
        if os.path.exists(os.path.join(save_dir, 'eval_log_aggregated.json')):
            print(f"Skip exist {save_dir}")
            continue
        print(f"Loaded model: {cfg.model_path}")
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0, batch_size=batch_size)
        dataset = datasets.load_dataset(cfg.data_path, split)['train']
        all_titles = list(set(dataset['title']))
        
        results, stats = {}, {}
        for mode in mode_to_key:
            if os.path.exists(os.path.join(save_dir, f'eval_log_{mode}.json')):
                print(f"Skip exist {save_dir}", mode)
                continue
            if mode == 'forget':
                load_split = split
            elif mode == 'hard_retain':
                load_split = f'{split}_{mode}'
            else:
                load_split = mode
            dataset = datasets.load_dataset(cfg.data_path, load_split)['train']
            results[load_split] = {'generated_text': [], 'RougeL': [], 'raw_generated_text': []}
            inputs = ['<s>' + unlearn_prompt.format(entity=all_titles, question=item['question'], type=target_type) for item in dataset]
            outputs = pipe(inputs, do_sample=False, max_new_tokens=max_new_tokens)
            for i, out in enumerate(outputs):
                generated_text = out[0]['generated_text'].split('[/INST]')[-1].strip()
                results[load_split]['raw_generated_text'].append(generated_text)
                results[load_split]['generated_text'].append([dataset[i]['question'], generated_text, dataset[i]['answer']])
                scores = scorer.score(dataset[i]['answer'], generated_text)
                results[load_split]['RougeL'].append(scores['rougeL'].recall)
            
            stats[f'{mode_to_key[mode]} ROUGE'] = np.mean(results[load_split]['RougeL'])
            print(f"Split: {load_split}")
            print(f"RougeL: {np.mean(results[load_split]['RougeL'])}")
            with open(os.path.join(save_dir, f'eval_log_{mode}.json'), 'w') as f:
                json.dump(results[load_split], f, indent=2)
        
        with open(os.path.join(save_dir, 'eval_log_aggregated.json'), 'w') as f:
            json.dump(results, f, indent=2)
        df = pd.DataFrame([stats])
        df.to_csv(f'{save_dir}/aggregate_stat.csv', index=False)


if __name__ == '__main__':
    main()