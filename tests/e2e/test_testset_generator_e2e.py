import os
import pytest
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from ragas.testset import TestsetGenerator
import os

@pytest.fixture
def setup_environment():
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

@pytest.fixture
def processed_docs():
    return [
        Document(metadata={'title': 'Machine Learning', 'id': 1}, page_content='Machine learning is a field of artificial intelligence focused on enabling systems to learn patterns from data.'),
        Document(metadata={'title': 'Deep Learning', 'id': 2}, page_content='Deep learning is a subset of machine learning utilizing neural networks with many layers.'),
        Document(metadata={'title': 'Natural Language Processing (NLP)', 'id': 3}, page_content='NLP is a branch of AI that enables computers to understand, interpret, and generate human language.')
    ]

@pytest.fixture
def llm_and_embeddings():
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4-turbo"))
    generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    return generator_llm, generator_embeddings

def test_generate_eval_dataset(setup_environment, processed_docs, llm_and_embeddings):
    generator_llm, generator_embeddings = llm_and_embeddings
    generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
    
    eval_dataset = generator.generate(testset_size=3, with_debugging_logs=True)
    
    assert eval_dataset is not None
    assert len(eval_dataset) == 3  # Adjust based on your expectations 