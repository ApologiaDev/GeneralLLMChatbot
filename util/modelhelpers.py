
import boto3
from langchain import HuggingFaceHub
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.embeddings.fake import FakeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import CTransformers
from langchain_community.embeddings.gpt4all import GPT4AllEmbeddings
from langchain_aws.llms.bedrock import BedrockLLM


def get_llm_model(config):
    llm_config = config['llm']
    hub = llm_config.get('hub', 'openai')
    if hub == 'openai':
        model = llm_config.get('model', 'gpt-3.5-turbo')
        temperature = config.get('temperature', 0.7)
        max_tokens = config.get('max_tokens', 600)
        return ChatOpenAI(temperature=temperature, model_name=model, max_tokens=max_tokens)
    elif hub == 'huggingface':
        model = llm_config.get('model', 'google/flan-t5-xxl')
        model_kwargs = llm_config.get('model_kwargs')  # example: {'temperature': temperature, 'max_length': max_tokens}
        if model_kwargs is None:
            return HuggingFaceHub(repo_id=model)
        else:
            return HuggingFaceHub(repo_id=model, model_kwargs=model_kwargs)
    elif hub == 'ctransformers':
        model = llm_config.get('model', 'TheBloke/Llama-2-7b-Chat-GGUF')
        model_file = llm_config.get('model_file', 'llama-2-7b-chat.Q2_K.gguf')
        model_type = llm_config.get('type', 'llama')
        model_config = llm_config.get('model_kwargs')  # example: {'max_new_tokens': 512, 'temperature': 0.5, 'gpu_layers':50}
        return CTransformers(model=model, model_file=model_file, type=model_type, config=model_config)
    else:
        raise ValueError('Unknown LLM specified!')


def get_embeddings_model(config):
    embedding_config = config.get('embedding')
    if embedding_config is None:
        return FakeEmbeddings(size=768)
    hub = embedding_config.get('hub')
    if (hub is None) or (hub == 'openai'):
        return OpenAIEmbeddings()
    elif hub == 'huggingface':
        model = embedding_config.get('model', 'sentence-transformers/all-MiniLM-L6-v2')
        model_kwargs = embedding_config.get('model_kwargs')
        embeddings_model = HuggingFaceEmbeddings(model_name=model) if model_kwargs is None else HuggingFaceEmbeddings(model_name=model, model_kwargs=model_kwargs)
        if embeddings_model.client.tokenizer.pad_token is None:
            embeddings_model.client.tokenizer.pad_token = embeddings_model.client.tokenizer.eos_token
        return embeddings_model
    elif hub == 'gpt4all':
        return GPT4AllEmbeddings()


def get_bedrock_runtime(region_name, *args, **kwargs):
    return boto3.client(service_name='bedrock-runtime', region_name=region_name, *args, **kwargs)


def get_langchain_bedrock_llm(model_id, client, *args, **kwargs):
    return BedrockLLM(model_id=model_id, client=client, *args, **kwargs)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=20,
    length_function=len
)