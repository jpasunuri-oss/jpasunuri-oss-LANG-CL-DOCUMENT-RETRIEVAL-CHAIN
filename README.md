# Overview

## Document Retrieval Chains
- A common use-case for LangChain chains is using documents to augment queries to an LLM, often this will be done in the case of Retrieval Augmented Generation (RAG).
- Documents can be loaded in a chain to facilitate giving access to their content to an LLM in order to provide it with the necessary information to produce a correct response

# Document Retrieval Chains Lab

## Files to Modify
The code you should modify, as well as the instructions for completing the lab, are in ```src/main/lab.py```. The lab module contains a main section you may use to manually test your code. You should not modify ```src/tests/lab_test.py```. You must pass all test cases provided in tests/lab_test.py to pass the lab.

A module with an example document loading chain and the steps taken to produce the chain can be found in ```src/main/app.py```. Use the example as a guide for completing the lab

## Notes and Resources
LangChain has an example of how to use document loading to augment a query to an LLM [here](https://python.langchain.com/docs/expression_language/get_started#rag-search-example)