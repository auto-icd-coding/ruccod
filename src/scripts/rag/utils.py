import re
import json
import ast
import logging

from pathlib import Path


def parse_ann_files(path, n=None, with_diagnosis2icd_per_file=False):
    """
    Parse .ann files to extract diagnoses and ICD codes.

    Parameters:
    - path (str or Path): Path to a directory containing .ann files or a single .ann file.
    - n (int, optional): Number of files to process if a directory is provided.
                         Defaults to None, meaning all files will be processed.

    Returns:
    - dict: A dictionary with diagnoses as keys and corresponding ICD codes as values.
    """
    diagnosis_dict = {}
    diagnosis2icd_per_file = []
    path = Path(path)

    # Helper function to parse a single .ann file
    def parse_file(filepath):
        entity_map = {}
        file_diagnosis2icd = {}
        with filepath.open("r", encoding="utf-8") as file:
            for line in file:
                # Match entities
                entity_match = re.match(r"^T(\d+)\s+icd_code\s+\d+\s+\d+\s+(.+)", line)
                if entity_match:
                    entity_id = f"T{entity_match.group(1)}"
                    diagnosis = entity_match.group(2).strip().lower().rstrip(".,?!")
                    if diagnosis:
                        entity_map[entity_id] = diagnosis

                # Match references and extract ICD codes
                reference_match = re.match(
                    r"^N\d+\s+Reference\s+(T\d+)\s+ICD_codes:\d+\s+([A-Z]\S+)", line
                )
                if reference_match:
                    entity_id = reference_match.group(1)
                    icd_code_text = reference_match.group(2).strip()
                    if entity_id in entity_map:
                        diagnosis = entity_map[entity_id]
                        if diagnosis not in diagnosis_dict:
                            diagnosis_dict[diagnosis] = icd_code_text
                        file_diagnosis2icd.update({diagnosis: icd_code_text})
        diagnosis2icd_per_file.append(file_diagnosis2icd)

    # Check if path is a directory or single file
    if path.is_dir():
        file_paths = sorted(path.glob("*.ann"), key=lambda x: int(x.stem))
        for i, filepath in enumerate(file_paths):
            if n is not None and i >= n:
                break
            parse_file(filepath)
    elif path.is_file() and path.suffix == ".ann":
        parse_file(path)

    if with_diagnosis2icd_per_file:
        return diagnosis_dict, diagnosis2icd_per_file
    return diagnosis_dict


def parse_txt_files(path, n=None):
    text_list = []
    path = Path(path)

    def parse_file(filepath):
        with filepath.open("r", encoding="utf-8") as file:
            file_content = file.read()
        return file_content

    if path.is_dir():
        file_paths = sorted(path.glob("*.txt"), key=lambda x: int(x.stem))
        for i, filepath in enumerate(file_paths):
            if n is not None and i >= n:
                break
            text_list.append(parse_file(filepath))
        return text_list
    elif path.is_file() and path.suffix == ".txt":
        return parse_file(path)
    return None


def parse_llm_response_to_list(response):
    try:
        response = response.strip()
        list_pattern = re.compile(r"\[.*\]")
        match = list_pattern.search(response)
        if match:
            list_str = match.group(0)
            parsed_list = ast.literal_eval(list_str)
            return parsed_list
        else:
            print("No valid list found in the response")
            return None
    except (ValueError, SyntaxError):
        print("Invalid Python list format of response")
        return None


def extract_first_answer_number(text):
    match = re.search(r"\d+", text)
    return int(match.group()) if match else None


def generate_ann_file(text, entities, entity_to_icd, output_filename="output.ann"):
    """
    Generate an .ann file from a list of entities and a dictionary mapping entities to ICD codes.

    :param entities: List of entities (e.g., ['Аднексит ', 'Кисты экзоцервикса'])
    :param entity_to_icd: Dictionary mapping entities to their ICD codes (e.g., {'аднексит': 'N70.9', 'кисты экзоцервикса': 'N88.8'})
    :param output_filename: Name of the output .ann file (default: "output.ann")
    """

    def _find_substring_indices(text, substr):
        start_index = text.find(substr)
        if start_index == -1:
            return (0, 0)
        end_index = start_index + len(substr)
        return (start_index, end_index)

    with open(output_filename, "w", encoding="utf-8") as file:
        if entities:
            for idx, entity in enumerate(entities, start=1):
                clean_entity = entity.strip().lower()

                start_index, end_index = _find_substring_indices(text, entity)

                t_line = f"T{idx}\ticd_code {start_index} {end_index}\t{entity.strip()}\n"
                file.write(t_line)

                icd_code = entity_to_icd.get(clean_entity, "UNK")

                n_line = f"N{idx}\tReference T{idx} ICD_codes:0000\t{icd_code}\n"
                file.write(n_line)


def save_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False)


def load_json(file_path):
    with open(file_path) as file:
        return json.load(file)


def initialize_logger(log_file_path, logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
