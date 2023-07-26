
import os
import json
from argparse import ArgumentParser
from glob import glob

from dotenv import load_dotenv
from langchain import HuggingFaceHub
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS


# load environment variables from .env, tokens
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAIKEY')
os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv('HUGGINGFACEHUB_API_TOKEN')


def get_argparser():
    argparser = ArgumentParser(description='Learn the context.')
    argparser.add_argument('corpusdir', help='directory of the PDF books')
    argparser.add_argument('learnconfigpath', help='path of the JSON file with the configs')
    argparser.add_argument('outputdir', help='target directory')
    return argparser


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
        return HuggingFaceHub(repo_id=model, model_kwargs=model_kwargs)
    else:
        raise ValueError('Unknown LLM specified!')


def get_embedding_model(config):
    embedding_config = config['embedding']
    hub = embedding_config.get('hub')
    if hub is None:
        return OpenAIEmbeddings()
    elif hub == 'huggingface':
        model = embedding_config.get('model', 'sentence-transformers/all-MiniLM-L6-v2')
        model_kwargs = embedding_config.get('model_kwargs')
        return HuggingFaceEmbeddings(model_name=model, model_kwargs=model_kwargs)


text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=900,
    chunk_overlap=20,
    length_function=len
)


def get_pages_from_pdf_document(pdffilepath):
    loader = PyPDFLoader(pdffilepath)
    pages = loader.load_and_split(text_splitter=text_splitter)
    return pages


def get_pages_from_pdf_documents(directory):
    pages = []
    for pdffilepath in glob(os.path.join(directory, '*.pdf')):
        this_pages = get_pages_from_pdf_document(pdffilepath)
        if this_pages is not None and len(this_pages) > 0:
            pages.append(this_pages)
    return pages


if __name__ == '__main__':
    args = get_argparser().parse_args()
    config = json.load(open(args.learnconfigpath, 'r'))
    if not os.path.isdir(args.outputdir):
        raise FileNotFoundError('Output directory {} does not exist.'.format(args.outputdir))

    llm = get_llm_model(config)
    embedding = get_embedding_model(config)

    pages = get_pages_from_pdf_documents(args.corpusdir)

    db = FAISS.from_documents(pages, embedding)

    db.save_local(args.outputdir)
