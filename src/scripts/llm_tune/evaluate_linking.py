import os, sys

BASE_PATH = ''

sys.path.append(os.path.join(BASE_PATH))
sys.path.append(os.path.join(BASE_PATH, 'src'))
#os.chdir(BASE_PATH)

import re
import yaml
import typing as t
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from transformers import GenerationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM, PeftConfig, PeftModel

from datasets import LinkingDataset
from config.config_eval import linking_subtask
from config.config_eval import actual_models, linking_icd_checkpoints, linking_train_and_icd_checkpoints
from config.config_eval import gen_config, batch_size, linking_icd_datasets, linking_train_and_icd_datasets


CONF = {'seq_max_length': 768}

def correct_tokens(tokens_tensor: np.array, tokenizer):
    tokens_tensor[tokens_tensor == 0] = 2 # convert output id 0 to 2 (eos_token_id)
    tokens_tensor[tokens_tensor < 0] = 1 # convert improper tokens to ''
    tokens_tensor[tokens_tensor > tokenizer.vocab_size] = 2
    return tokens_tensor

def prepate_output(output: np.array, tokenizer: AutoTokenizer) -> t.List[str]:
    #assert output.dim() == 2
    output = correct_tokens(output, tokenizer)
    output_strs = tokenizer.batch_decode(output, skip_special_tokens=True)
    return output_strs

def generate_batch(data, tokenizer, model, generation_config):
    """
    Генерация ответов для батча данных.

    Args:
        data: Словарь, содержащий тензоры ввода (input_ids, attention_mask и т.д.).
        tokenizer: Токенизатор.
        model: Модель.
        generation_config: Конфигурация генерации.

    Returns:
        Список сгенерированных текстов.
    """
    with torch.inference_mode():
        data = {k: v.to(model.device) for k, v in data.items()}
        output_ids = model.generate(**data, generation_config=generation_config)

    output_texts = prepate_output(output_ids.detach().cpu().numpy(), tokenizer)
    
    # Постобработка: извлечение ответа
    processed_texts = []
    for output_text in output_texts:
        try:
            output_text = output_text.split('Ответ:')[1][:2]
            output_text = output_text.strip().strip('\n')
            processed_texts.append(output_text)
        except IndexError:
            processed_texts.append("") # or handle this case differently

    return processed_texts

def load_tokenizer_and_model(model_name, checkpoint) -> t.Tuple[AutoTokenizer, AutoPeftModelForCausalLM]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    config = PeftConfig.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        return_dict=True,
        device_map="auto",
        attn_implementation='eager',
        trust_remote_code=True,
        #torch_dtype=torch.float16,
        torch_dtype=torch.bfloat16,
        #load_in_8bit=True,
        #load_in_4bit= True
    )
    model = PeftModel.from_pretrained(model, checkpoint)

    return tokenizer, model

def get_save_path(model_name: str) -> str:

    model_adapters_path = os.path.join(
        f'results/linking_preds_v2_bergamot/{linking_subtask}', 
        model_name
    )
    if not os.path.exists(model_adapters_path):
        os.makedirs(model_adapters_path)

    save_path = os.path.join(
        model_adapters_path,
        f'{linking_subtask}.parquet'
    )
    return save_path

def main():

    if linking_subtask == 'icd':
        checkpoints = linking_icd_checkpoints
        datasets = linking_icd_datasets
    elif linking_subtask == 'train_and_icd':
        checkpoints = linking_train_and_icd_checkpoints
        datasets = linking_train_and_icd_datasets

    for model_name, checkpoint, ds_path in zip(actual_models, checkpoints, datasets):

        oper_test_df = pd.read_parquet(ds_path)

        print(f'loading {model_name} best checkpoint')
        tokenizer, model = load_tokenizer_and_model(model_name, checkpoint)

        test_ds = LinkingDataset(oper_test_df, tokenizer, CONF, 'test')

        gen_config["eos_token_id"] = tokenizer.eos_token_id
        generation_config = GenerationConfig.from_dict(gen_config)

        results_ds = []
        for i in tqdm(range(0, len(test_ds), batch_size)):
            batch_data = [test_ds[j] for j in range(i, min(i + batch_size, len(test_ds)))]

            # Преобразование списка словарей в словарь списков
            batch_data_dict = {}
            for key in batch_data[0].keys():
                batch_data_dict[key] = torch.stack([d[key] for d in batch_data])

            batch_texts = oper_test_df.iloc[i:min(i + batch_size, len(test_ds))]['initial_text'].tolist()
            batch_entities = oper_test_df.iloc[i:min(i + batch_size, len(test_ds))]['mkb_text'].tolist()
            predicted_indxs_list = generate_batch(batch_data_dict, tokenizer, model, generation_config)

            for text, entity, predicted_idx in zip(batch_texts, batch_entities, predicted_indxs_list):
                results_ds.append({
                    'text': text,
                    'entity': entity,
                    'predicted_idx': predicted_idx
                })

        results_df = pd.DataFrame(results_ds)
        results_df.to_parquet(get_save_path(model_name))

if __name__ == '__main__':
    main()
