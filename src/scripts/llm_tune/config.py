actual_models = [
    'microsoft/Phi-3.5-mini-instruct',
    'Qwen/Qwen2.5-7B-Instruct',
    'm42-health/Llama3-Med42-8B',
    'mistralai/Mistral-Nemo-Instruct-2407',
]

LO_RA_CONFIG = {
    'r': 16, 
    'lora_alpha': 16, 
    'lora_dropout': 0.1, 
    'bias': "none", 
    #'modules_to_save': ["classifier"], 
    'task_type': "CAUSAL_LM",
}

DEVICE = "cuda"
LR = 5e-5
BATCH_SIZE = 2
GRAD_ACC = int(16 / BATCH_SIZE)
NUM_EPOCHS = 33

#TASK_TYPE = 'ner'
TASK_TYPE = 'linking'

RETRIEVAL_DS = 'icd'
#RETRIEVAL_DS = 'icd_and_train'

if TASK_TYPE == 'ner':
    TRAIN_DATASET = 'data/oper_train_df_v2.parquet'
    TEST_DATASET = 'data/oper_test_df_v2.parquet'
elif TASK_TYPE == 'linking':
    if RETRIEVAL_DS == 'icd':
        TRAIN_DATASET = 'data/retrieval_ds/train_icd_retrieval_df_stratified_bergamot.parquet'
        TEST_DATASET = 'data/retrieval_ds/test_icd_retrieval_df_stratified_bergamot.parquet'
    elif RETRIEVAL_DS == 'icd_and_train':
        TRAIN_DATASET = 'data/retrieval_ds/train_tr_and_icd_retrieval_df_stratified.parquet'
        TEST_DATASET = 'data/retrieval_ds/test_tr_and_icd_retrieval_df_stratified.parquet'