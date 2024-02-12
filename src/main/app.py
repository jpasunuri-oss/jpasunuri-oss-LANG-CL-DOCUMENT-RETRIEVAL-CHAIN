import os
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.document_loaders import TextLoader
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser



"""
    This module has an example chain that loads relevant information from a document and uses the
    content of the document to augment the query made to the llm in the chain. Each step in producing
    the chain is explained below.
    
    The chain is going to need four fundamental pieces:
        - search and retrieval
        - formatted prompt
        - llm functionality
        - response output
"""

"""
    Search and Retrieval:
        here is where the chain will load relevant information from a document. Since this this example
        uses an ephemeral vector store the document will need to be loaded first. 
"""

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
EMBED_ENDPOINT = os.getenv("EMBED_ENDPOINT")
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT")

#  create embedding function for our Chroma client
huggingface_embedding_function = HuggingFaceInferenceAPIEmbeddings(api_key=HUGGINGFACEHUB_API_TOKEN, api_url=EMBED_ENDPOINT)

# create our Chroma client
chroma_client = Chroma(embedding_function=huggingface_embedding_function, collection_name="text")

# load a text file
lab_text = TextLoader("src/resources/lab_txt.txt").load()

# load lab_text into vector store
chroma_client.add_documents(lab_text)

# create a VectorStoreRetriever object for search and retrieval in the chain
# this tool is the tool we need to use for search and retrieval in our chain: Note that we only
# have access to this method because we went through langchain to create our Chroma client.
text_retriever = chroma_client.as_retriever()

# LangChain provides the means of sending the user query to the retriever object in order to get
# the relevant documents, so we are done with setting up the first part of the chain.

"""
    Now that we have our retriever we need to setup the prompt. In this example we are going to make use
    of LangChain's ChatPromptTemplate. We are going to use the template below for our prompt:
"""

# the template
template = """answer the question based only on the following context:
{context}

Question: 
{question}
"""

# we pass the template to the ChatPromptTemplate method to create the prompt object our chain
prompt = ChatPromptTemplate.from_template(template)

"""
    Now we will setup the llm configuration: since we are using a HuggingFace endpoint we need to make
    a HuggingFaceEndpoint object
"""
# here we configure the llm we are using via HuggingFace
llm = HuggingFaceEndpoint(
    endpoint_url=LLM_ENDPOINT,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    task="text2text-generation",
    model_kwargs={
        "max_new_tokens": 200
    }
)

"""
    All the pieces are setup we can now create the chain. Our chain will have four steps to follow:
        1. send the user query to the retriever to get any relevant documents and send them to the prompt
        2. set the context question of the prompt before sending it to the llm
        3. query the llm with our prompt and send the response to the string output
        4. observe the response from the LLM
"""

# the parenthesis are just for implicit line continuation
text_chain = (
    {"context": text_retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


results = text_chain.invoke("how many books do Americans read per year?")
print(results) # the response from the LLM should say something about sources conflicting, some say 5, some 12