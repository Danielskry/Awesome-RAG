# 😎 Awesome Retrieval Augmented Generation (RAG) [![Awesome](https://awesome.re/badge-flat.svg)](https://awesome.re)

This repository contains a curated [Awesome List](https://github.com/sindresorhus/awesome) and general information on Retrieval-Augmented Generation (RAG) applications in Generative AI.

Retrieval-Augmented Generation (RAG) is an architectural approach aimed at providing additional context to Large Language Models (LLMs). It is commonly employed to complement foundational pre-trained LLMs with information that is either up-to-date, sensitive, or supplementary and specific. This approach enhances the model's ability to generate more accurate and contextually relevant outputs by integrating external facts and knowledge.

If we were to draw an analogy, we could liken RAG to an exam where we're permitted to consult specific textbooks. In contrast, typical large language models serve as foundational and all-purpose generative models, akin to showing up for the exam without a specific textbook but rather with just their extensive knowledge. This highlights how RAG enriches the generative process by allowing the model to access and integrate relevant external information, akin to consulting additional resources during an exam to enhance performance and accuracy.

## Content

- [ℹ️ General Information on RAG](#ℹ%EF%B8%8F-general-information-on-rag)
- [📖 Recommended Reading](#-recommended-reading)
- [💼 RAG Use Cases](#-rag-use-cases)
- [🧰 Frameworks that Facilitate RAG](#-frameworks-that-facilitate-rag)
- [📄 Embeddings](#-embeddings)
- [💾 Databases](#-databases)
- [📚 RAG papers](#-rag-papers)


## ℹ️ General Information on RAG

In traditional RAG approaches, we typically employ a basic architecture capable of retrieving a number of documents to enrich the context of a prompt for an LLM. This is generally achieved by retrieving documents that correspond to the input provided to the LLM prompt. For example, if we inquire about suitable materials for renovating a room in our apartment, the LLM may possess extensive information on room renovation, tools and associated materials. However, a foundational LLM may lack specific knowledge about our room, necessitating the gathering of additional context by referring to a blueprint of our room. Therefore, an RAG architecture might be employed to take our input regarding renovating our room and swiftly conduct a similarity search to match documents related to our question about renovation. If there is a match on documents related to our prompt, they will be used as additional context for the LLM to provide answers regarding renovation and materials specific to our room.

However, there is no guarantee that the similarity search will match documents based on the input, or that the LLM will be able to utilize the additional context autonomously. We may therefore, sometimes, need to adopt more advanced approaches for RAG that surpass mere naivety, such as integrating corrective measures, executing actions, and implementing iterative steps with the LLM before providing an answer. These elements can all be components of a more intricate RAG architecture, which may include:

- Implementing a [Corrective RAG](https://arxiv.org/pdf/2401.15884.pdf) (CRAG) approach.
- Employing [Retrieval-Augmented Fine-Tuning](https://techcommunity.microsoft.com/t5/ai-ai-platform-blog/raft-a-new-way-to-teach-llms-to-be-better-at-rag/ba-p/4084674) (RAFT) for additional enhancement.
- Incorporating [Reason and Action (ReAct)](https://research.google/blog/react-synergizing-reasoning-and-acting-in-language-models/) into the response generation process.
- Developing a [Self Reflective RAG](https://selfrag.github.io/).
- Performing a [RAG Fusion](https://arxiv.org/abs/2402.03367).
- Using [function/tool calling](https://python.langchain.com/docs/modules/model_io/chat/function_calling/) during inference.
- [Temporal Augmented Retrieval](https://adam-rida.medium.com/temporal-augmented-retrieval-tar-dynamic-rag-ad737506dfcc) (TAR)

## 📖 Recommended Reading

...

## 💼 RAG Use Cases

...

## 🧰 Frameworks that Facilitate RAG

- [LangChain](https://python.langchain.com/docs/modules/data_connection/) - An all-purpose framework for working with LLMs.
- [LlamaIndex](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/) - Framework for connecting custom data sources to LLMs.

## 📄 Embeddings

## 💾 Databases
The list below features several database systems suitable for Retrieval Augmented Generation (RAG) applications. They cover a range of RAG use cases, aiding in the efficient storage and retrieval of vectors to generate responses or recommendations.

**Distributed Data Processing and Serving Engines:**
- [Apache Cassandra](https://cassandra.apache.org/doc/latest/cassandra/vector-search/concepts.html): Distributed NoSQL database management system.
- [MongoDB Atlas](https://www.mongodb.com/products/platform/atlas-vector-search): Globally distributed, multi-model database service with integrated vector search.
- [Vespa](https://vespa.ai/): Open-source big data processing and serving engine designed for real-time applications.

**Search Engines with Vector Capabilities:**
- [Elasticsearch](https://www.elastic.co/elasticsearch): Provides vector search capabilities along with traditional search functionalities.
- [OpenSearch](https://github.com/opensearch-project/OpenSearch): Distributed search and analytics engine, forked from Elasticsearch.

**Vector Databases:**
- [Chroma DB](https://github.com/chroma-core/chroma): An AI-native open-source embedding database.
- [Milvus](https://github.com/milvus-io/milvus): An open-source vector database for AI-powered applications.
- [Pinecone](https://www.pinecone.io/): A serverless vector database, optimized for machine learning workflows.

**Relational Database Extensions:**
- [Pgvector](https://github.com/pgvector/pgvector): An open-source extension for vector similarity search in PostgreSQL.

**Other Database Systems:**
- [Azure Cosmos DB](https://learn.microsoft.com/en-us/azure/cosmos-db/vector-database): Globally distributed, multi-model database service with integrated vector search.
- [Couchbase](https://www.couchbase.com/products/vector-search/): A distributed NoSQL cloud database.
- [Lantern](https://lantern.dev/): A privacy-aware personal search engine.
- [LlamaIndex](https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores/): Employs a straightforward in-memory vector store for rapid experimentation.
- [Neo4j](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/): Graph database management system.
- [Qdrant](https://github.com/neo4j/neo4j): An open-source vector database designed for similarity search.
- [Redis Stack](https://redis.io/docs/latest/develop/interact/search-and-query/): An in-memory data structure store used as a database, cache, and message broker.
- [SurrealDB](https://github.com/surrealdb/surrealdb): A scalable multi-model database optimized for time-series data.
- [Weaviate](https://github.com/weaviate/weaviate): A open-source cloud-native vector search engine.

## 📚 RAG papers

- [Lewis, Patrick, et al. "Retrieval-augmented generation for knowledge-intensive nlp tasks." Advances in Neural Information Processing Systems 33 (2020): 9459-9474.](https://arxiv.org/pdf/2005.11401.pdf) - Paper that introduced RAG.

- [Gao, Yunfan, et al. "Retrieval-augmented generation for large language models: A survey." arXiv preprint arXiv:2312.10997 (2023)](https://arxiv.org/abs/2312.10997) - A survey of RAG for LLMs.
