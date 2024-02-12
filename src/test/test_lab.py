import unittest
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables.base import RunnableSequence
from src.main.lab import create_csv_chain, set_up_prompt_template, setup_csv_retriever, set_up_llm
from src.utilities.llm_testing_util import llm_connection_check, llm_wakeup, classify_relevancy


class TestDocumentLoadChain(unittest.TestCase):
    
    def test_setup_csv_retriever(self):
        retriever = setup_csv_retriever()
        self.assertEqual(type(retriever), VectorStoreRetriever)
        
    
    def test_setup_prompt_template(self):
        prompt = set_up_prompt_template()
        self.assertEqual(type(prompt), ChatPromptTemplate)
    
    def test_set_up_llm(self):
        llm = set_up_llm()
        self.assertEqual(type(llm), HuggingFaceEndpoint)
    
    def test_create_csv_chain(self):
        chain = create_csv_chain()
        self.assertEqual(type(chain), RunnableSequence)
        
    def test_run_chain(self):
        chain = create_csv_chain()
        response = chain.invoke("Suggest a good action movie to watch")
        options = [
            "The Dark Knight", 
            "Inception", 
            "Star Wars: Episode V - The Empire Strikes Back", 
            "The Matrix"
            ]
        has_relevant_suggestion = False
        for option in options:
            if option in response:
                has_relevant_suggestion = True
                break
        print(response.strip())
        self.assertTrue(has_relevant_suggestion)
        
class TestLLMResponses(unittest.TestCase):

    """
    This function is a sanity check for the Language Learning Model (LLM) connection.
    It attempts to generate a response from the LLM. If a 'Bad Gateway' error is encountered,
    it initiates the LLM wake-up process. This function is critical for ensuring the LLM is
    operational before running tests and should not be modified without understanding the
    implications.
    Raises:
        Exception: If any error other than 'Bad Gateway' is encountered, it is raised to the caller.
    """
    def test_llm_sanity_check(self):
        try:
            response = llm_connection_check()
            self.assertIsInstance(response, LLMResult)
        except Exception as e:
            if 'Bad Gateway' in str(e):
                llm_wakeup()
                self.fail("LLM is not awake. Please try again in 3-5 minutes.")