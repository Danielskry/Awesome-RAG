# 😎 Awesome Retrieval Augmented Generation (RAG) [![Awesome](https://awesome.re/badge-flat.svg)](https://awesome.re)

This repository contains a curated Awesome List and general information on the topic of RAG for Large Language Models (LLMs).

Retrieval-Augmented Generation (RAG) is an architectural approach aimed at providing additional context to Large Language Models (LLMs). It is commonly employed to complement foundational pre-trained LLMs with information that is either up-to-date, sensitive, or supplementary and specific. This approach enhances the model's ability to generate more accurate and contextually relevant outputs by integrating external facts and knowledge.

If we were to draw an analogy, we could liken RAG to an exam where we're permitted to consult specific textbooks. In contrast, typical large language models serve as foundational and all-purpose generative models, akin to showing up for the exam without a specific textbook but rather with just their extensive knowledge. This highlights how RAG enriches the generative process by allowing the model to access and integrate relevant external information, akin to consulting additional resources during an exam to enhance performance and accuracy.

## Content

- [General Information on RAG](#general-information-on-rag)
- [Recommended Reading](#recommended-reading)
- [RAG Use Cases](#rag-use-cases)
- [Frameworks that Facilitate RAG](#frameworks-that-facilitate-rag)
- [Embedding & Vector databases](#embedding--vector-databases)
- [RAG papers](#rag-papers)


## General Information on RAG

In traditional RAG approaches, we typically employ a basic architecture capable of retrieving a number of documents to enrich the context of a prompt for an LLM. This is generally achieved by retrieving documents that correspond to the input provided to the LLM prompt. For example, when we inquire about suitable materials for renovating a room in our apartment, the LLM may possess extensive information on room renovation and associated materials. However, a foundational LLM may lack specific knowledge about our room, necessitating the gathering of additional context by referring to a blueprint of our room. Therefore, an RAG architecture might be employed to take our input regarding renovating our room and swiftly conduct a similarity search to match documents related to our question about renovation. If there is a match on documents related to our prompt, they will be used as additional context for the LLM to provide answers regarding renovation and materials specific to our room.

However, there is no guarantee that the similarity search will match documents based on the input, or that the LLM will be able to utilize the additional context autonomously. Therefore, we may sometimes need to adopt more advanced approaches for RAG that surpass mere naivety, such as integrating corrective measures, executing actions, and implementing iterative steps with the LLM before providing an answer. These elements can all be components of a more intricate RAG architecture, which may include:

- Implement a [Corrective RAG](https://arxiv.org/pdf/2401.15884.pdf) (CRAG) approach.
- Employ [Retrieval-Augmented Fine-Tuning](https://techcommunity.microsoft.com/t5/ai-ai-platform-blog/raft-a-new-way-to-teach-llms-to-be-better-at-rag/ba-p/4084674) (RAFT) for additional enhancement.
- Incorporate [Reason and Action (ReAct)](https://research.google/blog/react-synergizing-reasoning-and-acting-in-language-models/) into the response generation process.
- Develop a [Self Reflective RAG](https://selfrag.github.io/).
- Perform [RAG Fusion](https://arxiv.org/abs/2402.03367).
- Use [function/tool calling](https://python.langchain.com/docs/modules/model_io/chat/function_calling/) during inference.

## Recommended Reading

...

## RAG Use Cases

...

## Frameworks that Facilitate RAG

- [LangChain](https://python.langchain.com/docs/modules/data_connection/) - An all-purpose framework for working with LLMs.
- [LlamaIndex](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/) - Framework for connecting custom data sources to LLMs.

## RAG Embedding & Vector Databases

...

## RAG papers

- [Lewis, Patrick, et al. "Retrieval-augmented generation for knowledge-intensive nlp tasks." Advances in Neural Information Processing Systems 33 (2020): 9459-9474.](https://arxiv.org/pdf/2005.11401.pdf) - Introduced RAG.
