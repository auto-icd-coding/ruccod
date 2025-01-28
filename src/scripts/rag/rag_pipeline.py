import argparse
from pathlib import Path

from models_config import load_model_and_embeddings, llm_invoke
from db_config import (
    load_train_dataset,
    load_test_dataset,
    load_db,
    db_similarity_search,
)
from utils import (
    parse_llm_response_to_list,
    extract_first_answer_number,
    generate_ann_file,
    initialize_logger,
)
from prompts import (
    generate_entity_extraction_messages,
    generate_entity_matching_messages,
)

DATASETS_DIR = Path("./datasets")
DATASETS = [
    "mkb-full",
    "mkb-full-ehr-train",
    "ruccon",
    "nerel-bio",
]
MODELS = {
    "hf": [
        "microsoft/Phi-3.5-mini-instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "m42-health/Llama3-Med42-8B",
        "mistralai/Mistral-Nemo-Instruct-2407",
    ],
    "ollama": [
        "llama3.1:8b-instruct-fp16",
        # "llama3.1:70b-instruct-fp16"
    ],
}
EMBEDDINGS = ["BAAI/bge-m3", "linking_model_bergamot_biosyn_ep19"]
PROVIDERS = ["hf", "ollama"]


def pipeline(dataset):
    test_dir = DATASETS_DIR / "test"

    diagnosis2icd, diagnoses = load_train_dataset(dataset)
    texts_test, diagnosis2icd_per_file_test = load_test_dataset(test_dir)
    number_diagnosis2icd_to_retrieve = sum(len(icds) for icds in diagnosis2icd_per_file_test)

    for provider in PROVIDERS:
        for embedding_name in EMBEDDINGS:
            db_dir = Path(f"./db/{dataset}-diagnoses-{embedding_name.replace('/', '-')}")
            for model_name in MODELS[provider]:
                results_dir = Path(
                    f"./results/{dataset}/{embedding_name.replace('/', '-')}/{model_name.replace('/', '-')}"
                )
                results_dir.mkdir(parents=True, exist_ok=True)

                log_file = results_dir / "log.log"
                logger = initialize_logger(
                    log_file, logger_name=f"{dataset}_{embedding_name}_{model_name}"
                )

                llm, embeddings = load_model_and_embeddings(provider, model_name, embedding_name)
                logger.info(f"Model loaded: {model_name}")

                db = load_db(db_dir, diagnoses, embeddings)
                logger.info(f"DB loaded: {embedding_name}")

                success_retrieves = 0

                for index, text in enumerate(texts_test):
                    logger.info(f"[{index}] Text: {text}")

                    response = llm_invoke(
                        llm, generate_entity_extraction_messages, text, provider=provider
                    )
                    logger.info(f"Extraction response: {response}")

                    extracted_diagnoses = parse_llm_response_to_list(response)
                    logger.info(f"Extracted diagnoses: {extracted_diagnoses}")

                    diagnoses_found = []
                    if extracted_diagnoses:
                        extracted_diagnoses = [
                            extracted_diagnosis.lower()
                            for extracted_diagnosis in extracted_diagnoses
                            if extracted_diagnosis
                        ]
                        for possible_diagnosis in extracted_diagnoses:
                            possible_diagnosis = possible_diagnosis.rstrip(" ?,.!")
                            logger.info(f"For possible diagnonsis: {possible_diagnosis}")

                            retrieved_diagnoses_list, retrieved_diagnoses_dict = (
                                db_similarity_search(possible_diagnosis, db)
                            )
                            logger.info(f"---- Retrieved: {retrieved_diagnoses_dict}")

                            final_diagnosis = None
                            possible_icd = diagnosis2icd.get(possible_diagnosis, None)
                            if possible_icd is not None:
                                retrieved_icds = [
                                    diagnosis2icd[d] for d in retrieved_diagnoses_list
                                ]
                                if possible_icd in retrieved_icds:
                                    success_retrieves += 1
                                    logger.info("---- Success diagnosis retrieve")
                                    if possible_diagnosis in retrieved_diagnoses_list:
                                        final_diagnosis = possible_diagnosis
                                    else:
                                        logger.warning(
                                            "---- ---- Failed to find possible diagnosis in retrieved"
                                        )

                            if final_diagnosis is None:
                                response = llm_invoke(
                                    llm,
                                    generate_entity_matching_messages,
                                    possible_diagnosis,
                                    retrieved_diagnoses_dict,
                                    provider=provider,
                                )
                                logger.info(f"---- ---- Matching response: {response}")
                                final_diagnosis_number = extract_first_answer_number(response)
                                if final_diagnosis_number is None:
                                    logger.warning(
                                        "---- ---- Failed to extract diagnosis number from response"
                                    )
                                    final_diagnosis_number = 1
                                if (
                                    final_diagnosis_number > len(retrieved_diagnoses_list)
                                    or final_diagnosis_number < 1
                                ):
                                    logger.warning(
                                        f"---- ---- Failed to extract diagnosis number in the range 1 to {len(retrieved_diagnoses_list)}"
                                    )
                                    final_diagnosis_number = 1
                                final_diagnosis = retrieved_diagnoses_dict[final_diagnosis_number]

                            logger.info(f"---- Final diagnosis: {final_diagnosis}")
                            diagnoses_found.append(final_diagnosis)
                    else:
                        logger.warning("Failed to extract any diagnoses from text")
                    generate_ann_file(
                        text.lower(),
                        diagnoses_found,
                        diagnosis2icd,
                        output_filename=results_dir / f"{index}.ann",
                    )
                    logger.info(
                        f"---- Found ICD codes: {[diagnosis2icd[diagnosis] for diagnosis in diagnoses_found]} ({results_dir}/{index}.ann)"
                    )
                    logger.info(
                        f"---- Refer ICD codes: {list(diagnosis2icd_per_file_test[index].values())} ({test_dir}/{index}.ann)"
                    )
                    logger.info("-" * 42)
                logger.info(
                    f"Success retrieve: {success_retrieves / number_diagnosis2icd_to_retrieve * 100:.2f}%."
                )


def main():
    parser = argparse.ArgumentParser(description="Run a pipeline with a selected dataset.")
    parser.add_argument("dataset", choices=DATASETS, help=f"The dataset to use {DATASETS}")
    args = parser.parse_args()

    pipeline(args.dataset)


if __name__ == "__main__":
    main()
