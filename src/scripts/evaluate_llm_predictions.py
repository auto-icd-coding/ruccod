import argparse
import logging
import os.path
import re
from typing import Dict, List, Set

ANNOTATION_FILENAME_PATTERN = r"(?P<file_id>[0-9]+).ann"
ENTITY_ID_PATTERN = r"(?P<letter>[TN])(?P<number>[0-9]+)"


class Entity:
    def __init__(self, e_id, spans, e_type, entity_str, doc_id, concept_id_name="icd_code"):
        self.e_id = e_id
        self.spans = spans
        assert concept_id_name in ("icd_code", "digital_code")
        self.concept_id_name = concept_id_name
        # self.span_start = span_start
        # self.span_end = span_end
        self.e_type = e_type
        self.entity_str = entity_str
        self.icd_codes = None
        self.doc_id = doc_id

    def __str__(self):
        min_span = min(int(t[0]) for t in self.spans)
        max_span = min(int(t[1]) for t in self.spans)
        # max_span = max((int(t[1]) for t in self.spans))
        icd_codes = self.icd_codes
        for code in icd_codes:
            assert '|' not in code
        icd_codes = '|'.join(icd_codes)

        # assert '|' not in str(concept_id)

        return f"{self.e_id}||{min_span}|{max_span}||{self.e_type}||{self.entity_str}||{icd_codes}"

    def to_biosyn_str(self):
        assert '||' not in self.e_id
        assert '||' not in self.e_type
        assert '||' not in self.entity_str

        return self.__str__()


def read_brat_annotation_file(input_path: str) -> List[Entity]:
    entity_id2entity: Dict[str, Entity] = {}

    """
    T1	icd_code 1 47	ллергический ? острый   неуточненный  дерматит
    N1	Reference T1 ICD_codes:7625	L23
    T1	icd_code 0 19	Ушибленная гематома
    N1	Reference T1 ICD_codes:12772	T81.0

    T1 \t icd_code 1 47 \t ллергический ? острый   неуточненный  дерматит
    N1	Reference T1 ICD_codes:7625	L23
    T1\t icd_code 0 19 \t Ушибленная гематома
    N1 \t Reference T1 ICD_codes:12772 \t T81.0
    """

    # entity_id2entity_type: Dict[str, str] = {}
    # entity_id2cui: Dict[str, str] = {}

    with open(input_path, 'r', encoding="utf-8") as inp_file:
        doc_id = os.path.basename(input_path).split('.')[0]
        for line in inp_file:

            attrs = line.strip().split('\t')
            entity_id = attrs[0]
            m = re.fullmatch(ENTITY_ID_PATTERN, entity_id)

            if m is None:
                print(f"INVALID LINE FORMAT:\n\t{input_path}, {line.strip()}")
                continue
            # assert m is not None
            entity_letter = m.group("letter")
            # Processing mention spans
            if entity_letter == "T":
                assert len(attrs) == 3
                entity_type_and_span = attrs[1]
                e_t_s_split = entity_type_and_span.split()
                entity_type = e_t_s_split[0]

                spans = [t.split() for t in entity_type_and_span[len(entity_type):].strip().split(';')]
                # assert len(e_t_s_split) == 3

                entity_type = e_t_s_split[0]
                assert entity_type == "icd_code"

                mention_string = attrs[2]
                entity = Entity(e_id=entity_id, spans=spans, e_type=entity_type, entity_str=mention_string,
                                doc_id=doc_id)
                entity_id2entity[entity_id] = entity

            # Processing the linking to dictionary
            elif entity_letter == "N":
                assert len(attrs) == 3
                rel_type_and_entity_id_and_concept_id = attrs[1]

                rt_eid_cid_split = rel_type_and_entity_id_and_concept_id.split()

                assert len(rt_eid_cid_split) == 3
                rel_type = rt_eid_cid_split[0]
                assert rel_type == "Reference"
                entity_id = rt_eid_cid_split[1]

                vocab_concept_id = rt_eid_cid_split[2]

                assert vocab_concept_id.startswith("ICD_codes:")
                norm_digital_code = vocab_concept_id.lstrip("ICD_codes:")
                # assert entity_id2entity.get(entity_id) is not None
                if entity_id2entity.get(entity_id) is not None:
                    # entity_id2entity[entity_id].digital_code = norm_digital_code
                    icd_code = attrs[2]
                    if entity_id2entity[entity_id].icd_codes is None:
                        entity_id2entity[entity_id].icd_codes = set()
                    entity_id2entity[entity_id].icd_codes.add(icd_code)

    entities = list(entity_id2entity.values())

    return entities


def read_brat_directory(directory: str) -> Dict[int, List[Entity]]:
    file_id2entities_list: Dict[int, List[Entity]] = {}
    logging.info(f"Processing directory: {directory}")
    for filename in os.listdir(directory):
        if not filename.endswith("ann"):
            continue
        m = re.fullmatch(ANNOTATION_FILENAME_PATTERN, filename)
        assert m is not None
        file_id = m.group("file_id")
        assert file_id2entities_list.get(file_id) is None

        file_path = os.path.join(directory, filename)
        entities_list = read_brat_annotation_file(input_path=file_path)
        file_id2entities_list[file_id] = entities_list

    logging.info(f"Finished processing directory: {directory}")

    return file_id2entities_list


def filter_invalid_ner_spans(file_id2entities_list: Dict[int, List[Entity]]):
    dropped_invalid_span_counter = 0
    dropped_code_less_counter = 0
    dropped_invalid_span_entities_list = []
    dropped_codeless_entities_list = []
    for file_id in file_id2entities_list.keys():
        entities_list = file_id2entities_list[file_id]
        filtered_entities_list = []

        for entity in entities_list:
            min_span = min(int(t[0]) for t in entity.spans)
            code = entity.icd_codes
            if code is None:
                dropped_code_less_counter += 1
                dropped_codeless_entities_list.append(entity)

            elif min_span != -1:
                filtered_entities_list.append(entity)
            else:
                dropped_invalid_span_counter += 1
                dropped_invalid_span_entities_list.append(entity)
        file_id2entities_list[file_id] = filtered_entities_list
    logging.info(f"Finished Dropping invalid entities (with left span equal to -1 and code-less)")
    logging.info(f"Dropped entities with invalid spans: {dropped_invalid_span_counter}")
    for e in dropped_invalid_span_entities_list[:5]:
        logging.info(str(e))
    logging.info(f"Dropped code-less entities: {dropped_code_less_counter}")
    for e in dropped_codeless_entities_list[:5]:
        logging.info(f"{e.doc_id}|{e.entity_str}")


def calculate_document_level_metrics(ref_file_id2entities_list: Dict[int, List[Entity]],
                                     pred_file_id2entities_list: Dict[int, List[Entity]]):
    eval_dict: Dict[str, float] = {}
    num_documents = 0
    doc_level_true_code_counter = 0
    doc_level_true_span_counter = 0

    span_tp_sum = 0
    span_fn_sum = 0
    span_fp_sum = 0

    code_tp_sum = 0
    code_fn_sum = 0
    code_fp_sum = 0

    span_code_tp_sum = 0
    span_code_fn_sum = 0
    span_code_fp_sum = 0

    for file_id in ref_file_id2entities_list.keys():
        num_documents += 1
        ref_entities_list = ref_file_id2entities_list[file_id]
        pred_entities_list = pred_file_id2entities_list[file_id]

        ref_codes: Set[str] = set()
        ref_spans: Set[str] = set()
        pred_codes: Set[str] = set()
        pred_spans: Set[str] = set()
        ref_span_codes: Set[str] = set()
        pred_span_codes: Set[str] = set()

        for entity in ref_entities_list:
            assert len(entity.icd_codes) == 1

            ref_code = list(entity.icd_codes)[0]
            ref_codes.add(ref_code)
            assert len(entity.spans) == 1
            min_span = min(int(t[0]) for t in entity.spans)
            max_span = min(int(t[1]) for t in entity.spans)
            ref_spans.add(f"{min_span}|{max_span}")
            ref_span_codes.add(f"{min_span}|{max_span}|{ref_code}")

        for entity in pred_entities_list:
            assert len(entity.icd_codes) == 1
            pred_code = list(entity.icd_codes)[0]
            pred_codes.add(pred_code)
            assert len(entity.spans) == 1
            min_span = min(int(t[0]) for t in entity.spans)
            max_span = min(int(t[1]) for t in entity.spans)
            pred_spans.add(f"{min_span}|{max_span}")
            pred_span_codes.add(f"{min_span}|{max_span}|{pred_code}")

        span_tp = len(pred_spans.intersection(ref_spans))
        span_fn = len(ref_spans.difference(pred_spans))
        span_fp = len(pred_spans.difference(ref_spans))

        code_tp = len(pred_codes.intersection(ref_codes))
        code_fn = len(ref_codes.difference(pred_codes))
        code_fp = len(pred_codes.difference(ref_codes))

        span_code_tp = len(pred_span_codes.intersection(ref_span_codes))
        span_code_fn = len(ref_span_codes.difference(pred_span_codes))
        span_code_fp = len(pred_span_codes.difference(ref_span_codes))

        span_tp_sum += span_tp
        span_fn_sum += span_fn
        span_fp_sum += span_fp

        code_tp_sum += code_tp
        code_fn_sum += code_fn
        code_fp_sum += code_fp

        span_code_tp_sum += span_code_tp
        span_code_fn_sum += span_code_fn
        span_code_fp_sum += span_code_fp

        if len(ref_codes.intersection(pred_codes)) == len(ref_codes):
            doc_level_true_code_counter += 1
        if len(ref_spans.intersection(pred_spans)) == len(ref_spans):
            doc_level_true_span_counter += 1

    eval_dict["Document-level NER accuracy"] = doc_level_true_span_counter / num_documents
    eval_dict["Document-level Code assignment accuracy"] = doc_level_true_code_counter / num_documents

    code_p, code_r, code_f, code_acc = \
        calculate_prfacc(code_tp_sum, code_fp_sum, code_fn_sum)
    span_p, span_r, span_f, span_acc = \
        calculate_prfacc(span_tp_sum, span_fp_sum, span_fn_sum)
    span_code_p, span_code_r, span_code_f, span_code_acc = \
        calculate_prfacc(span_code_tp_sum, span_code_fp_sum, span_code_fn_sum)
    eval_dict["NER precision"] = span_p
    eval_dict["NER recall"] = span_r
    eval_dict["NER F-score"] = span_f
    eval_dict["NER accuracy"] = span_acc

    eval_dict["NER+Linking precision"] = span_code_p
    eval_dict["NER+Linking recall"] = span_code_r
    eval_dict["NER+Linking F-score"] = span_code_f
    eval_dict["NER+Linking accuracy"] = span_code_acc

    eval_dict["Code assignment precision"] = code_p
    eval_dict["Code assignment recall"] = code_r
    eval_dict["Code assignment F-score"] = code_f
    eval_dict["Code assignment accuracy"] = code_acc

    return eval_dict


def calculate_prfacc(tp, fp, fn):
    # Precision
    pr_denom = tp + fp
    if pr_denom != 0:
        p = tp / pr_denom
    else:
        p = 0.
    # Recall
    r_denom = tp + fn
    if r_denom != 0:
        r = tp / r_denom
    else:
        r = 0.
    # F1-score
    f1_denom = p + r
    if f1_denom != 0:
        f1 = 2 * p * r / f1_denom
    else:
        f1 = 0.
    # Accuracy
    acc_denom = tp + fp + fn
    if acc_denom != 0:
        acc = tp / acc_denom
    else:
        acc = 0.

    return p, r, f1, acc


def main(args):
    ref_data_dir = args.ref_data_dir
    pred_data_dir = args.pred_data_dir
    output_path = args.output_path
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)

    ref_file_id2entities_list = read_brat_directory(ref_data_dir)
    pred_file_id2entities_list = read_brat_directory(pred_data_dir)

    logging.info("Filtering reference dataset...")
    filter_invalid_ner_spans(ref_file_id2entities_list)
    logging.info("Filtering prediction dataset...")
    filter_invalid_ner_spans(pred_file_id2entities_list)

    eval_metrics_dict = calculate_document_level_metrics(ref_file_id2entities_list=ref_file_id2entities_list,
                                                         pred_file_id2entities_list=pred_file_id2entities_list)
    logging.info("Finished evaluation")
    print("Finished evaluation")
    for k, v in eval_metrics_dict.items():
        logging.info(f"{k}\t{v}")
        print(f"{k}\t{v}")
    with open(output_path, 'w+', encoding="utf-8") as out_file:
        for k, v in eval_metrics_dict.items():
            out_file.write(f"{k}\t{v}\n")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = argparse.ArgumentParser()

    parser.add_argument('--ref_data_dir', type=str)
    parser.add_argument('--pred_data_dir', type=str)
    parser.add_argument('--output_path', type=str)
    arguments = parser.parse_args()
    main(arguments)
