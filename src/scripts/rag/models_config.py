import transformers
import torch
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFacePipeline,
    HuggingFaceEmbeddings,
)
from langchain_ollama import ChatOllama


API_KEYS = {
    "giga": "",
    "openai": "",
}


def load_model_and_embeddings(provider, model_name, embedding_name=None):
    """
    Load model and embeddings based on provider selection.
    :param provider: "huggingface", "ollama"
    :param model_name: model identifier string for the chosen provider
    :param embedding_name: embedding identifier (optional for Hugging Face and Ollama)
    :return: tuple (llm, embeddings)
    """

    if provider == "hf":
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            max_new_tokens=2048,
            # do_sample=False,
        )
        pipeline.tokenizer.pad_token_id = pipeline.tokenizer.eos_token_id
        model_kwargs = {
            "temperature": 0.0,
        }
        llm = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipeline, model_kwargs=model_kwargs))
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False},
        )
    elif provider == "ollama":
        llm = ChatOllama(
            model=model_name,
            temperature=0.0,
            num_predict=2048,
        )
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False},
        )
    else:
        raise ValueError(
            "Invalid provider specified. Choose 'huggingface', 'ollama', 'giga', or 'openai'."
        )

    return llm, embeddings


def llm_invoke(llm, message_generator, *args, provider=None, **kwargs):
    """Invoke the LLM with the given message generator and handle provider-specific logic."""
    response = llm.invoke(message_generator(*args, **kwargs))
    response = response.content
    if provider == "hf":
        prompt = llm.tokenizer.apply_chat_template(
            message_generator(*args, hf=True, **kwargs),
            tokenize=False,
            add_generation_prompt=True,
        )
        response = response[len(prompt) :]
    return response
