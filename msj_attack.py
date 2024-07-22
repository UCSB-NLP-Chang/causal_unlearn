import json
import os

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets
from rouge_score import rouge_scorer
import numpy as np
import pandas as pd
import hydra

from utils import unlearn_prompt


@hydra.main(version_base=None, config_path="config", config_name="forget_wpu")
def main(cfg):
    attack_budget = 100
    max_generate_tokens = 4096
    batch_size = 4
    files = os.listdir(cfg.save_dir)
    files = [f for f in files if f.startswith('checkpoint-')]
    save_dir = os.path.join(cfg.save_dir, files[0])
    if os.path.exists(os.path.join(save_dir, 'msj_attack_output.json')):
        print(f"Skip exist {save_dir}")
        return
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
    tokenizer.padding_side = "left"
    tokenizer.padding_size = 'longest'
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model_path = "meta-llama/Llama-2-7b-chat-hf" if cfg.forget_loss == 'prompt_baseline' else cfg.save_dir
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
    print(f"Loaded model: {model_path}")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0, batch_size=batch_size)
    dataset = datasets.load_dataset(cfg.data_path, cfg.split)['train']
    all_titles = list(set(dataset['title']))
    target_type = 'people'
    
    with open('data/msj_attack.json', 'r') as f:
        attack_ices = json.load(f)
    attack_ices = attack_ices[-attack_budget:]
    for i, item in enumerate(attack_ices):
        attack_ices[i]['generated'] = attack_ices[i]['generated'].split('\n\n')[0].split('[/INST]')[-1].strip()
    attack_ices = [f"[INST] {item['question']} [/INST] {item['generated']}" for item in attack_ices]
    attack_query = '</s><s>'.join(attack_ices)[7:-7].strip()
    # Reduce incontext example length
    max_query_token = 3896
    if cfg.forget_loss == 'prompt_baseline':
        if '100' in cfg.split:
            max_query_token -= 800
        else:
            max_query_token -= 296
    while len(tokenizer(attack_query)['input_ids']) > max_query_token:
        attack_budget -= 1
        attack_query = '</s><s>'.join(attack_ices[-attack_budget:])[7:-7].strip()
    
    results = {'generated_text': [], 'RougeL': [], 'raw_generated_text': []}
    inputs = []
    for item in dataset:
        question = f"{attack_query}</s><s>[INST] {item['question']}"
        if cfg.forget_loss == 'prompt_baseline':
            prompt = '<s>' + unlearn_prompt.format(entity=all_titles, question=question, type=target_type)
        else:
            prompt = f"<s>[INST] {question} [/INST]"
        inputs.append(prompt)
    outputs = pipe(inputs, do_sample=False, max_length=max_generate_tokens)
    for i, out in enumerate(outputs):
        generated_text = out[0]['generated_text'].split('[/INST]')[-1].strip()
        results['raw_generated_text'].append(generated_text)
        results['generated_text'].append([dataset[i]['question'], generated_text, dataset[i]['answer']])
        scores = scorer.score(dataset[i]['answer'], generated_text)
        results['RougeL'].append(scores['rougeL'].recall)

    print(f"Split: {cfg.split}")
    print(f"RougeL: {np.mean(results['RougeL'])}")
    with open(os.path.join(save_dir, 'msj_attack_output.json'), 'w') as f:
        json.dump(results, f, indent=2)
    df = pd.read_csv(f'{save_dir}/aggregate_stat.csv')
    df[f'MSJ attack ROUGE'] = np.mean(results['RougeL'])
    df.to_csv(f'{save_dir}/aggregate_stat.csv', index=False)


if __name__ == '__main__':
    print("###############################")
    print("Many-shot jailbreaking attack")
    print("###############################")
    main()