from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from utils import load_json, parse_ann_files, parse_txt_files

RETRIEVE_K = 15


def load_train_dataset(dataset_name):
    train_file_path = Path(f"./datasets/{dataset_name}-diagnosis2icd.json")
    diagnosis2icd = load_json(train_file_path)
    diagnoses = list(diagnosis2icd.keys())
    return diagnosis2icd, diagnoses


def load_test_dataset(test_dir):
    _, diagnosis2icd_per_file_test = parse_ann_files(test_dir, with_diagnosis2icd_per_file=True)
    texts_test = parse_txt_files(test_dir)
    return texts_test, diagnosis2icd_per_file_test


def load_db(
    file_path,
    content,
    embeddings,
    distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
    normalize=False,
):
    vectorstore_path = Path(file_path)
    if vectorstore_path.exists():
        db = FAISS.load_local(
            vectorstore_path,
            embeddings,
            allow_dangerous_deserialization=True,
            distance_strategy=distance_strategy,
            normalize_L2=normalize,
        )
    else:
        db = FAISS.from_texts(
            content,
            embedding=embeddings,
            distance_strategy=distance_strategy,
            normalize_L2=normalize,
        )
        db.save_local(vectorstore_path)
        db = FAISS.load_local(
            vectorstore_path,
            embeddings,
            allow_dangerous_deserialization=True,
            distance_strategy=distance_strategy,
            normalize_L2=normalize,
        )
    return db


def db_similarity_search(query, db, k=RETRIEVE_K):
    result = db.similarity_search_with_score(query, k)
    result_list = [item[0].page_content for item in result]
    result_dict = {index + 1: item for index, item in enumerate(result_list)}
    return result_list, result_dict
