from data_module import TextForgetDatasetQA, TextForgetDatasetQADistill
from dataloader import CustomTrainerForgetting, custom_data_collator_forget, custom_data_collator_distill
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed

import hydra 
import transformers
import os
from peft import LoraConfig, get_peft_model, PeftModel
from pathlib import Path
from utils import get_model_identifiers_from_yaml
from omegaconf import OmegaConf


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


@hydra.main(version_base=None, config_path="config", config_name="forget_wpu")
def main(cfg):
    num_devices = int(os.environ.get('WORLD_SIZE', 1))
    print(f"num_devices: {num_devices}")

    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))

    set_seed(cfg.seed)
    print(f"seed: {cfg.seed}")

    os.environ["WANDB_DISABLED"] = "true"
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    if cfg.model_path is None:
        cfg.model_path = model_cfg["ft_model_path"]

    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    # save cfg in cfg.save_dir
    if local_rank == 0:
        with open(f"{cfg.save_dir}/config.yaml", "w") as file:
            OmegaConf.save(cfg, file)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if cfg.input_type == "question":
        max_length = 256 if 'TOFU' in cfg.data_path else 512
    else:
        if cfg.forget_loss in ["intervention", 'prompt_distill', 'whp']:
            max_length = 400
        else:
            max_length = 3072
    print('='*20 + f"Max length: {max_length}" + '='*20)
    if cfg.forget_loss in ["intervention", 'prompt_distill', 'whp', 'di']:
        torch_format_dataset = TextForgetDatasetQADistill(cfg, tokenizer=tokenizer, max_length=max_length)
    else:
        torch_format_dataset = TextForgetDatasetQA(cfg.data_path, tokenizer=tokenizer, model_family = cfg.model_family, max_length=max_length, split=cfg.split, loss_type=cfg.forget_loss, input_type=cfg.input_type)
    
    batch_size = cfg.batch_size
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    steps_per_epoch = len(torch_format_dataset)//(batch_size*gradient_accumulation_steps*num_devices)
    max_steps = int(cfg.num_epochs*len(torch_format_dataset)) // (batch_size*gradient_accumulation_steps*num_devices)
    if 'TOFU' in cfg.data_path:
        save_strategy = 'steps'
        save_steps = steps_per_epoch
        eval_steps = max_steps + 1 # separate evaluation
    else:
        save_strategy = 'no'
        save_steps = max_steps + 1
        eval_steps = max_steps
    
    print("######################")
    print("Saving to: ", cfg.save_dir)
    print("######################")

    if any([os.path.exists(os.path.join(cfg.save_dir, f'checkpoint-{i}')) for i in [cfg.num_epochs, steps_per_epoch*cfg.num_epochs]]):
        print("Directory already exists")
        if not cfg.overwrite_dir:
            exit()

    print(f"batch_size per device: {batch_size}")
    print(f"gradient_accumulation_steps: {gradient_accumulation_steps}")
    print(f"max_steps: {max_steps}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Eval steps: {eval_steps}")
    print(f"Weight decay: {cfg.weight_decay}")
    print('='*20 + "Data sample shape" + '='*20)
    if local_rank == 0:
        for ind, each in enumerate(torch_format_dataset[0]):
            if isinstance(each[0], list):
                for t in each[0]:
                    print('='*20 + f"Decoded example {tokenizer.decode(t)}" + '='*20)
            else:
                print('='*20 + f"Decoded example {tokenizer.decode(each[0])}" + '='*20)

    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=max(1, max_steps//10),
        save_strategy=save_strategy,
        save_only_model=True,
        max_steps=max_steps,
        learning_rate=cfg.lr,
        bf16=True,
        bf16_full_eval=True,
        logging_steps=max(1,max_steps//20),
        logging_dir=f'{cfg.save_dir}/logs',
        output_dir=cfg.save_dir,
        optim="paged_adamw_32bit",
        save_steps=save_steps,
        ddp_find_unused_parameters= False,
        deepspeed='config/ds_config.json',
        weight_decay=cfg.weight_decay,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        seed=cfg.seed
    )

    oracle_model = None
    
    print('='*20 + f"Loading from checkpoint {cfg.model_path}" + '='*20)
    config = AutoConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_path, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, trust_remote_code = True)
    if cfg.forget_loss in ["KL", 'npo']:
        oracle_model = AutoModelForCausalLM.from_pretrained(cfg.model_path, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, trust_remote_code = True)
    
    #now we have a HuggingFace model 
    if model_cfg["gradient_checkpointing"] == "true":
        model.gradient_checkpointing_enable()
    trainable_modules = find_all_linear_names(model)
    config = LoraConfig(
        r=cfg.LoRA.r, 
        lora_alpha=cfg.LoRA.alpha, 
        target_modules=trainable_modules, 
        lora_dropout=cfg.LoRA.dropout,
        bias="none", 
        task_type="CAUSAL_LM"
    )
    if cfg.LoRA.r != 0:
        model = get_peft_model(model, config)
        print_trainable_parameters(model)
        print(trainable_modules)

    data_collator = custom_data_collator_distill if cfg.forget_loss in ["intervention", 'prompt_distill', 'whp', 'di'] else custom_data_collator_forget
    trainer = CustomTrainerForgetting(
        model=model,
        tokenizer=tokenizer,
        train_dataset=torch_format_dataset,
        eval_dataset = torch_format_dataset,
        compute_metrics=None,                # the callback for computing metrics, None in this case since you're doing it in your callback
        # callbacks=[GlobalStepDeletionCallback],
        args=training_args,
        data_collator=data_collator,
        oracle_model = oracle_model,
        forget_loss = cfg.forget_loss,
        eval_cfg = cfg.eval,
        retain_strength = cfg.retain_strength,
        beta = cfg.beta,
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()
    
    # save the model
    if cfg.LoRA.r != 0:
        model = model.merge_and_unload()

    if 'TOFU' not in cfg.data_path:
        trainer.save_model(cfg.save_dir)

    #delete all "global_step*" files in the save_dir/checkpoint-*/ directories
    if local_rank == 0:
        for file in Path(cfg.save_dir).glob("checkpoint-*"):
            for global_step_dir in file.glob("global_step*"):
                #delete the directory
                import shutil
                shutil.rmtree(global_step_dir)


if __name__ == "__main__":
    main()
