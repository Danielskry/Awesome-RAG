# 😎 Awesome Retrieval Augmented Generation (RAG) [![Awesome](https://awesome.re/badge-flat.svg)](https://awesome.re)

This repository contains a curated [Awesome List](https://github.com/sindresorhus/awesome) and general information on Retrieval-Augmented Generation (RAG) applications in Generative AI.

Retrieval-Augmented Generation (RAG) is a technique in Generative AI where additional context is retrieved from external sources to enrich the generative process of Large Language Models (LLMs). This approach allows LLMs to incorporate up-to-date, specific, or sensitive information that they may lack from their pre-training data alone.

## Content

- [ℹ️ General Information on RAG](#ℹ%EF%B8%8F-general-information-on-rag)
- [🎯 Approaches](#-approaches)
- [💼 RAG Use Cases](#-rag-use-cases)
- [🧰 Frameworks that Facilitate RAG](#-frameworks-that-facilitate-rag)
- [🛠️ Techniques](#-techniques)
- [📊 Metrics](#-metrics)
- [💾 Databases](#-databases)
- [📚 RAG papers](#-rag-papers)


## ℹ️ General Information on RAG
In traditional RAG approaches, a basic framework is employed to retrieve documents that enrich the context of an LLM prompt. For instance, when querying about materials for renovating a house, the LLM may possess general knowledge about renovation but lacks specific details about the particular house. Implementing an RAG architecture allows for quick searching and retrieval of relevant documents, such as blueprints, to offer more customized responses. This ensures that the LLM incorporates specific information to the renovation needs, thereby enhancing the accuracy of its responses.

## 🎯 Approaches

RAG implementations vary in complexity, from simple document retrieval to advanced techniques integrating iterative feedback loops and domain-specific enhancements. Approaches may include:

- [Data cleaning techniques](https://medium.com/intel-tech/four-data-cleaning-techniques-to-improve-large-language-model-llm-performance-77bee9003625): Pre-processing steps to refine input data and improve model performance.
- [Corrective RAG](https://arxiv.org/pdf/2401.15884.pdf) (CRAG): Methods to correct or refine the retrieved information before integration into LLM responses.
- [Retrieval-Augmented Fine-Tuning](https://techcommunity.microsoft.com/t5/ai-ai-platform-blog/raft-a-new-way-to-teach-llms-to-be-better-at-rag/ba-p/4084674) (RAFT): Techniques to fine-tune LLMs specifically for enhanced retrieval and generation tasks.
- [Reason and Action (ReAct)](https://research.google/blog/react-synergizing-reasoning-and-acting-in-language-models/) (ReAct): Integration of reasoning capabilities to guide LLM responses based on retrieved context.
- [Self Reflective RAG](https://selfrag.github.io/): Models that dynamically adjust retrieval strategies based on model performance feedback.
- [RAG Fusion](https://arxiv.org/abs/2402.03367): Techniques combining multiple retrieval methods for improved context integration.
- [Temporal Augmented Retrieval](https://adam-rida.medium.com/temporal-augmented-retrieval-tar-dynamic-rag-ad737506dfcc) (TAR): Considering time-sensitive data in retrieval processes.
- [Plan-then-RAG](https://arxiv.org/abs/2406.12430) (PlanRAG): Strategies involving planning stages before executing RAG for complex tasks.
- [Tagging and Labeling](https://python.langchain.com/v0.1/docs/use_cases/tagging/): Adding semantic tags or labels to retrieved data to enhance relevance.

## 💼 RAG Use Cases

...

## 🧰 Frameworks that Facilitate RAG

- [LangChain](https://python.langchain.com/docs/modules/data_connection/) - An all-purpose framework for working with LLMs.
- [Semantic Kernel](https://github.com/microsoft/semantic-kernel) - An SDK from Microsoft for developing Generative AI applications.
- [LlamaIndex](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/) - Framework for connecting custom data sources to LLMs.
- [Cognita](https://github.com/truefoundry/cognita) - Open-source RAG framework for building modular and production ready applications.

## 🛠️ Techniques

### Chunking
- **[Fixed-size chunking](https://medium.com/@anuragmishra_27746/five-levels-of-chunking-strategies-in-rag-notes-from-gregs-video-7b735895694d)**
  - Dividing text into consistent-sized segments for efficient processing.
  - Splits texts into chunks based on size and overlap.
  - Example: [Split by character](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/character_text_splitter/) (LangChain).
  - Example: [SentenceSplitter](https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/sentence_splitter/) (LlamaIndex).
- **[Recursive chunking](https://medium.com/@AbhiramiVS/chunking-methods-all-to-know-about-it-65c10aa7b24e)**
  - Hierarchical segmentation using recursive algorithms for complex document structures.
  - Example: [Recursively split by character](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/) (LangChain).
- **[Document-based chunking](https://medium.com/@david.richards.tech/document-chunking-for-rag-ai-applications-04363d48fbf7)**
  - Segmenting documents based on metadata or formatting cues for targeted analysis.
  - Example: [MarkdownHeaderTextSplitter](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/markdown_header_metadata/) (LangChain).
  - Example: Handle image and text embeddings with models like [OpenCLIP](https://github.com/mlfoundations/open_clip).
- **[Semantic chunking](https://www.youtube.com/watch?v=8OJC21T2SL4&t=1933s)**
  - Extracting meaningful sections based on semantic relevance rather than arbitrary boundaries.
- **[Agentic chunking](https://youtu.be/8OJC21T2SL4?si=8VnYaGUaBmtZhCsg&t=2882)**
  - Interactive chunking methods where LLMs guide segmentation. 

### Embeddings
- Select embedding model
  - **[MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)**: Explore Hugging Face's benchmark for evaluating model embeddings.
  - **Custom Embeddings**: Develop tailored embeddings for specific domains or tasks to enhance model performance.


## 📊 Metrics

### Search metrics

These metrics play a crucial role in evaluating how effectively RAG systems match and integrate external documents or data sources to enhance the context and accuracy of generated outputs.

- [Euclidean Distance](https://en.wikipedia.org/wiki/Euclidean_distance)
  - Measure of distance between vectors in Euclidean space.
  - Effective for dense and continuous feature vectors.
  - Often used in clustering algorithms like K-means.
- [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
  - Measure of similarity between vectors in a multi-dimensional space.
  - Commonly used in text and document similarity, such as when the direction of the vectors is more important than their magnitude.
- [Dot Product](https://en.wikipedia.org/wiki/Dot_product)
  - Measure of projection between vectors indicating similarity.
  - Simple and efficient, often used in combination with other methods.
- [Manhattan Distance (L1 Norm)](https://medium.com/swlh/different-types-of-distances-used-in-machine-learning-ec7087616442)
  - Sum of absolute differences between vector components.
  - Useful for grid-like data structures and when the differences are uniformly distributed.
- [Minkowski Distance](https://en.wikipedia.org/wiki/Minkowski_distance)
  - Generalized metric combining Euclidean and Manhattan distances.
  - Particularly effective for numerical datasets when comparing the similarity in magnitude among multiple data point vectors.
- [Jaccard Similarity](https://en.wikipedia.org/wiki/Jaccard_index)
  - Measure of overlap between sets.
  - Commonly used for binary or categorical data.
- [Hamming Distance](https://en.wikipedia.org/wiki/Hamming_distance)
  - Measure of difference between binary data sequences.
  - Used for binary strings or categorical data.
- [Mahalanobis Distance](https://en.wikipedia.org/wiki/Mahalanobis_distance)
  - Measure considering correlations in data distribution.
  - Valuable when the data has correlations.
- [Pearson Correlation Coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)
  - Measure of linear correlation between two variables.
  - Useful for understanding the linear relationship between variables.
 
### Evaluation metrics
These metrics assess the quality and relevance of generated answers, evaluating how accurate, contextually appropriate, and reliable they are for the intended applications.

...

## 💾 Databases
The list below features several database systems suitable for Retrieval Augmented Generation (RAG) applications. They cover a range of RAG use cases, aiding in the efficient storage and retrieval of vectors to generate responses or recommendations.

### Distributed Data Processing and Serving Engines:
- [Apache Cassandra](https://cassandra.apache.org/doc/latest/cassandra/vector-search/concepts.html): Distributed NoSQL database management system.
- [MongoDB Atlas](https://www.mongodb.com/products/platform/atlas-vector-search): Globally distributed, multi-model database service with integrated vector search.
- [Vespa](https://vespa.ai/): Open-source big data processing and serving engine designed for real-time applications.

### Search Engines with Vector Capabilities:
- [Elasticsearch](https://www.elastic.co/elasticsearch): Provides vector search capabilities along with traditional search functionalities.
- [OpenSearch](https://github.com/opensearch-project/OpenSearch): Distributed search and analytics engine, forked from Elasticsearch.

### Vector Databases:
- [Chroma DB](https://github.com/chroma-core/chroma): An AI-native open-source embedding database.
- [Milvus](https://github.com/milvus-io/milvus): An open-source vector database for AI-powered applications.
- [Pinecone](https://www.pinecone.io/): A serverless vector database, optimized for machine learning workflows.
- [Oracle AI Vector Search](https://www.oracle.com/database/ai-vector-search/#retrieval-augmented-generation): Integrates vector search capabilities within Oracle Database for semantic querying based on vector embeddings.

### Relational Database Extensions:
- [Pgvector](https://github.com/pgvector/pgvector): An open-source extension for vector similarity search in PostgreSQL.

### Other Database Systems:
- [Azure Cosmos DB](https://learn.microsoft.com/en-us/azure/cosmos-db/vector-database): Globally distributed, multi-model database service with integrated vector search.
- [Couchbase](https://www.couchbase.com/products/vector-search/): A distributed NoSQL cloud database.
- [Lantern](https://lantern.dev/): A privacy-aware personal search engine.
- [LlamaIndex](https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores/): Employs a straightforward in-memory vector store for rapid experimentation.
- [Neo4j](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/): Graph database management system.
- [Qdrant](https://github.com/neo4j/neo4j): An open-source vector database designed for similarity search.
- [Redis Stack](https://redis.io/docs/latest/develop/interact/search-and-query/): An in-memory data structure store used as a database, cache, and message broker.
- [SurrealDB](https://github.com/surrealdb/surrealdb): A scalable multi-model database optimized for time-series data.
- [Weaviate](https://github.com/weaviate/weaviate): A open-source cloud-native vector search engine.

### Vector Search Libraries and Tools:
- [FAISS](https://github.com/facebookresearch/faiss): A library for efficient similarity search and clustering of dense vectors, designed to handle large-scale datasets and optimized for fast retrieval of nearest neighbors.

## 📚 RAG papers

- [Lewis, Patrick, et al. "Retrieval-augmented generation for knowledge-intensive nlp tasks." Advances in Neural Information Processing Systems 33 (2020): 9459-9474.](https://arxiv.org/pdf/2005.11401.pdf) - Paper that introduced RAG.

- [Gao, Yunfan, et al. "Retrieval-augmented generation for large language models: A survey." arXiv preprint arXiv:2312.10997 (2023)](https://arxiv.org/abs/2312.10997) - A survey of RAG for LLMs.
