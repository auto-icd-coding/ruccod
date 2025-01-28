from langchain.schema import HumanMessage, SystemMessage


ENTITY_EXTRACTION_PROMPT = """Тебе будет предоставлен текст, содержащий диагнозы. Выдели диагнозы из этого текста. Не изменяй написание диагнозов в тексте. Ответь только в формате списка: ['диагноз1', 'диагноз2', ...]"""

ENTITY_MATCHING_PROMPT = """Тебе будет дан диагноз для референса, а также список диагнозов из базы данных. 
Твоя задача определить, какой диагноз из базы данных ближе всего подходит под диагноз референса. 
Старайся выбрать диагноз точно, обращая внимание на детали. Выбирай диагноз с наибольшим совпадением по словам и по смыслу. 
Ты можешь выбирать только из диагнозов из списка. 
Обращай больше внимания на диагнозы в начале списка, есть большая вероятность, что они подходят больше. 
Лучше выбрать более короткий диагноз, чем диагноз, содержащий информацию, которой нет в диагнозе для референса.
В ответе напиши только номер диагноза и ничего больше."""


def generate_entity_extraction_messages(text, hf=False):
    if hf:
        return [
            {"role": "system", "content": ENTITY_EXTRACTION_PROMPT},
            {"role": "user", "content": f"Текст:\n{text}"},
        ]
    return [
        SystemMessage(content=ENTITY_EXTRACTION_PROMPT),
        HumanMessage(content=f"Текст:\n{text}"),
    ]


def generate_entity_matching_messages(entity, diagnoses, hf=False):
    if hf:
        return [
            {"role": "system", "content": ENTITY_MATCHING_PROMPT},
            {
                "role": "user",
                "content": f"Диагноз для референса: {entity}\nСписок диагнозов из базы данных: {diagnoses}",
            },
        ]
    return [
        SystemMessage(content=ENTITY_MATCHING_PROMPT),
        HumanMessage(
            content=f"Диагноз для референса: {entity}\nСписок диагнозов из базы данных: {diagnoses}"
        ),
    ]
