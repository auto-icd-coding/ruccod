import os, sys

BASE_PATH = ''

sys.path.append(os.path.join(BASE_PATH))
sys.path.append(os.path.join(BASE_PATH, 'src'))
#os.chdir(BASE_PATH)

import typing as t
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, Dataset
from tqdm.notebook import tqdm

from transformers import EarlyStoppingCallback
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer

from datasets import NerDataset, LinkingDataset
from config.config import TASK_TYPE, TRAIN_DATASET, TEST_DATASET, RETRIEVAL_DS
from config.config import actual_models, LO_RA_CONFIG
from config.config import DEVICE, LR, BATCH_SIZE, GRAD_ACC, NUM_EPOCHS

def load_tokenizer_and_model(model_name: str) -> tuple[AutoTokenizer, AutoModelForCausalLM]:

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        use_fast=False,
        token=''
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        #torch_dtype=torch.float16,
        device_map=DEVICE,
        #attn_implementation='eager',
        token=''
    )

    lora_loaded = True
    for tm in [["q_proj", "v_proj"], ["q", "v"], ['qkv_proj']]:
        try:
            LO_RA_CONFIG['target_modules'] = tm
            peft_config = LoraConfig(**LO_RA_CONFIG)
            model = get_peft_model(model, peft_config)
            lora_loaded = True
            break
        except ValueError:
            continue
    
    if not lora_loaded:
        raise ValueError

    model = model.to(DEVICE)

    print(model.print_trainable_parameters())

    return tokenizer, model

def get_train_dataloader(self) -> DataLoader:
    return DataLoader(
        self.train_dataset,
        sampler=RandomSampler(self.train_dataset),
        batch_size=self.args.train_batch_size,
        pin_memory=True
    )
def get_eval_dataloader(self, eval_dataset) -> DataLoader:
    return DataLoader(
        self.eval_dataset,
        sampler=RandomSampler(self.eval_dataset),
        batch_size=self.args.eval_batch_size,
        pin_memory=True
    )

def train(
        model: AutoModelForCausalLM, 
        train_ds: NerDataset, 
        test_ds: NerDataset, 
        save_path: str
    ):
    args = TrainingArguments(
        save_path,
        remove_unused_columns=False,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=5,
        load_best_model_at_end=True,
        learning_rate=LR,
        warmup_ratio=0.1,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC,
        per_device_eval_batch_size=BATCH_SIZE,
        #fp16=True,
        bf16=True,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=10,
        label_names=["labels"],
        #fsdp="full_shard",
        metric_for_best_model='loss', 
        greater_is_better=False
    )

    Trainer.get_train_dataloader = get_train_dataloader
    Trainer.get_eval_dataloader = get_eval_dataloader

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        #compute_metrics=get_metrics_func(tokenizer),
        #preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[EarlyStoppingCallback(3, 0.0)],
    )
    trainer.train()

def get_save_path(model_name: str) -> str:

    model_adapters_path = os.path.join(
        f'models/{TASK_TYPE}_adapters', 
        model_name
    )
    if not os.path.exists(model_adapters_path):
        os.makedirs(model_adapters_path)

    already_exists_adapters = os.listdir(model_adapters_path)
    exist_indexes = [adapter_version.split('_')[-1] for adapter_version in already_exists_adapters]
    exist_indexes = [int(idx) for idx in exist_indexes]
    oper_index = 0 if not exist_indexes else max(exist_indexes) + 1

    save_path = os.path.join(
        model_adapters_path,
        f'{RETRIEVAL_DS}_v_{oper_index}'
    )
    os.makedirs(save_path)
    return save_path

def main():

    conf = {'seq_max_length': 768}

    oper_train_df = pd.read_parquet(TRAIN_DATASET)
    oper_test_df = pd.read_parquet(TEST_DATASET)

    for model_name in actual_models:
        print(f'train {model_name}')
        try:
            tokenizer, model = load_tokenizer_and_model(model_name)
        except ValueError:
            print(f'Not matched keys for model: {model_name}')
            continue

        ds_class = NerDataset if TASK_TYPE == 'ner' else LinkingDataset

        train_ds = ds_class(oper_train_df, tokenizer, conf, 'train')
        test_ds = ds_class(oper_test_df, tokenizer, conf, 'valid')
        print(len(train_ds), len(test_ds))

        save_path = get_save_path(model_name)
        print(f'adapters save path: {save_path}')

        train(model, train_ds, test_ds, save_path)

if __name__ == '__main__':
    main()
