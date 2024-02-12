"""
    In this lab you will be using LCEL to load documents from a vector store to augment queries to
    an llm (essentially creating mini Retrieval Augmented Generation apps). To complete this lab you
    will need to implement the methods called at each step of the two chains provided in this module. The
    first chain will load pdf content to help the LLM answer a question about the weather in Dallas, the 
    second chain will provide the LLM with movie context to help it answer a question about what movies
    to watch. 
    
    a main section has been provided for you to manually test your code, and the "app" module in the main
    package has an example of how to set up the various pieces for a chain to load documents for RAG
    operations.
    
    If you need further assistance LangChain has an example of how load documents from a vector store
    to augment queries to an llm here: https://python.langchain.com/docs/expression_language/get_started#rag-search-example
    Note that the LangChain example does not use Chroma or HuggingFace resources, which the lab does.
"""
import os

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
EMBED_ENDPOINT = os.getenv("EMBED_ENDPOINT")
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT")

def setup_csv_retriever():
    """
        use this function to set up a document retriever for the lab_csv file in the resources directory.
        You will need to code the following:
            - An embedding function for the Chroma client compatible with LangChain's HuggingFaceInferenceAPIEmbeddings
            - A Chroma client provided via LangChain
            - A CSV loader to load the lab csv
        Make sure to return a VectorStoreRetriever object that has access to the lab_csv document
    """
    # TODO: load "lab_csv.csv" into a Chroma vector store and return a Vectorstore retriever with the document
    pass

def set_up_prompt_template():
    """
        Use the provided template to create your promt for the LLM.
    """
    provided_template = """answer the question based only on the following context:
    {context}

    Question: 
    {question}
    """
    # TODO: create and return a chat template for the LLM
    pass

def set_up_llm():
    # TODO: create and return an HuggingFaceEndpoint object
    pass

def create_csv_chain():
    """
        TODO: use the previous methods in the lab to create the individual parts of the chain. Remember
        the order of operations:
            - search and retrieval
            - formatted prompt
            - llm functionality
            - response output
    """
    pass



if __name__ == "__main__":
    # Use this to manually test your code
    pass
    