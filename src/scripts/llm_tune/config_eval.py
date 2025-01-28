import os

actual_models = [
    'microsoft/Phi-3.5-mini-instruct',
    'Qwen/Qwen2.5-7B-Instruct',
    'm42-health/Llama3-Med42-8B',
    'mistralai/Mistral-Nemo-Instruct-2407',
]

models_path = ''
ner_adapters_path = os.path.join(models_path, 'ner_adapters')
linking_adapters_path = os.path.join(models_path, 'linking_adapters')

ner_checkpoints = [
    os.path.join(ner_adapters_path, 'microsoft/Phi-3.5-mini-instruct/v_0/checkpoint-1400'),
    os.path.join(ner_adapters_path, 'Qwen/Qwen2.5-7B-Instruct/v_0/checkpoint-1800'),
    os.path.join(ner_adapters_path, 'm42-health/Llama3-Med42-8B/v_0/checkpoint-1000'),
    os.path.join(ner_adapters_path, 'mistralai/Mistral-Nemo-Instruct-2407/v_0/checkpoint-1400'),
]
"""
linking_icd_checkpoints = [
    os.path.join(linking_adapters_path, 'microsoft/Phi-3.5-mini-instruct/v_0/checkpoint-1500'),
    os.path.join(linking_adapters_path, 'Qwen/Qwen2.5-7B-Instruct/v_0/checkpoint-1500'),
    os.path.join(linking_adapters_path, 'm42-health/Llama3-Med42-8B/v_0/checkpoint-1500'),
    os.path.join(linking_adapters_path, 'mistralai/Mistral-Nemo-Instruct-2407/v_0/checkpoint-1300'),
]

linking_train_and_icd_checkpoints = [
    os.path.join(linking_adapters_path, 'microsoft/Phi-3.5-mini-instruct/v_1/checkpoint-700'),
    os.path.join(linking_adapters_path, 'Qwen/Qwen2.5-7B-Instruct/v_1/checkpoint-700'),
    os.path.join(linking_adapters_path, 'm42-health/Llama3-Med42-8B/v_1/checkpoint-600'),
    os.path.join(linking_adapters_path, 'mistralai/Mistral-Nemo-Instruct-2407/v_1/checkpoint-700'),
]
"""
"""
linking_icd_checkpoints = [
    os.path.join(linking_adapters_path, 'microsoft/Phi-3.5-mini-instruct/icd_v_2/checkpoint-2800'),
    os.path.join(linking_adapters_path, 'Qwen/Qwen2.5-7B-Instruct/icd_v_2/checkpoint-2900'),
    os.path.join(linking_adapters_path, 'm42-health/Llama3-Med42-8B/icd_v_2/checkpoint-2700'),
    os.path.join(linking_adapters_path, 'mistralai/Mistral-Nemo-Instruct-2407/icd_v_2/checkpoint-2400'),
]

linking_train_and_icd_checkpoints = [
    os.path.join(linking_adapters_path, 'microsoft/Phi-3.5-mini-instruct/icd_and_train_v_3/checkpoint-2300'),
    os.path.join(linking_adapters_path, 'Qwen/Qwen2.5-7B-Instruct/icd_and_train_v_3/checkpoint-1400'),
    os.path.join(linking_adapters_path, 'm42-health/Llama3-Med42-8B/icd_and_train_v_3/checkpoint-1700'),
    os.path.join(linking_adapters_path, 'mistralai/Mistral-Nemo-Instruct-2407/icd_and_train_v_3/checkpoint-2100'),
]
"""
linking_icd_checkpoints = [
    os.path.join(linking_adapters_path, 'microsoft/Phi-3.5-mini-instruct/icd_v_4/checkpoint-1900'),
    os.path.join(linking_adapters_path, 'Qwen/Qwen2.5-7B-Instruct/icd_v_4/checkpoint-1900'),
    os.path.join(linking_adapters_path, 'm42-health/Llama3-Med42-8B/icd_v_4/checkpoint-2300'),
    os.path.join(linking_adapters_path, 'mistralai/Mistral-Nemo-Instruct-2407/icd_v_4/checkpoint-2400'),
]

linking_train_and_icd_checkpoints = []

gen_config = {
    #"max_new_tokens": 500,
    "max_new_tokens": 5,
    "pad_token_id": 1,
    #"no_repeat_ngram_size": 11,
    #"repetition_penalty": 1.04,
    "temperature": 0.01,
    #"top_p": 0.12,
    "do_sample": True,
}

batch_size = 32 # Выберите подходящий размер батча

ner_df_path = 'data/oper_test_df_v2.parquet'

linking_subtask = 'icd'
#linking_subtask = 'train_and_icd'
"""
linking_icd_datasets = [
    'data/preds_retrieval_ds/phi/icd_retrieval_df.parquet',
    'data/preds_retrieval_ds/qwen/icd_retrieval_df.parquet',
    'data/preds_retrieval_ds/llama/icd_retrieval_df.parquet',
    'data/preds_retrieval_ds/mistral/icd_retrieval_df.parquet'
]

linking_train_and_icd_datasets = [
    'data/preds_retrieval_ds/phi/tr_and_icd_retrieval_df.parquet',
    'data/preds_retrieval_ds/qwen/tr_and_icd_retrieval_df.parquet',
    'data/preds_retrieval_ds/llama/tr_and_icd_retrieval_df.parquet',
    'data/preds_retrieval_ds/mistral/tr_and_icd_retrieval_df.parquet'
]
"""
"""
linking_icd_datasets = [
    'data/preds_retrieval_ds/phi/icd_retrieval_df_stratified.parquet',
    'data/preds_retrieval_ds/qwen/icd_retrieval_df_stratified.parquet',
    'data/preds_retrieval_ds/llama/icd_retrieval_df_stratified.parquet',
    'data/preds_retrieval_ds/mistral/icd_retrieval_df_stratified.parquet'
]

linking_train_and_icd_datasets = [
    'data/preds_retrieval_ds/phi/tr_and_icd_retrieval_df_stratified.parquet',
    'data/preds_retrieval_ds/qwen/tr_and_icd_retrieval_df_stratified.parquet',
    'data/preds_retrieval_ds/llama/tr_and_icd_retrieval_df_stratified.parquet',
    'data/preds_retrieval_ds/mistral/tr_and_icd_retrieval_df_stratified.parquet'
]
"""
linking_icd_datasets = [
    'data/preds_retrieval_ds/phi/icd_retrieval_df_stratified_bergamot.parquet',
    'data/preds_retrieval_ds/qwen/icd_retrieval_df_stratified_bergamot.parquet',
    'data/preds_retrieval_ds/llama/icd_retrieval_df_stratified_bergamot.parquet',
    'data/preds_retrieval_ds/mistral/icd_retrieval_df_stratified_bergamot.parquet'
]
linking_train_and_icd_datasets = []