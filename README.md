# 😎 Awesome Retrieval Augmented Generation (RAG) [![Awesome](https://awesome.re/badge-flat.svg)](https://awesome.re)

This repository contains a curated [Awesome List](https://github.com/sindresorhus/awesome) and general information on Retrieval-Augmented Generation (RAG) applications in Generative AI.

Retrieval-Augmented Generation (RAG) is a technique in Generative AI where additional context is retrieved from external sources to enrich the generative process of Large Language Models (LLMs). This approach allows LLMs to incorporate up-to-date, specific, or sensitive information that they may lack from their pre-training data alone.

## Content

- [ℹ️ General Information on RAG](#ℹ%EF%B8%8F-general-information-on-rag)
- [🎯 Approaches](#-approaches)
- [🧰 Frameworks that Facilitate RAG](#-frameworks-that-facilitate-rag)
- [🛠️ Techniques](#-techniques)
- [📊 Metrics](#-metrics)
- [💾 Databases](#-databases)

## ℹ️ General Information on RAG

In traditional RAG approaches, a basic framework is employed to retrieve documents that enrich the context of an LLM prompt. For instance, when querying about materials for renovating a house, the LLM may possess general knowledge about renovation but lacks specific details about the particular house. Implementing an RAG architecture allows for quick searching and retrieval of relevant documents, such as blueprints, to offer more customized responses. This ensures that the LLM incorporates specific information to the renovation needs, thereby enhancing the accuracy of its responses.

**A typical RAG implementation follows these key steps:**

1. **Divide the knowledge base:** Break the document corpus into smaller, manageable chunks.
2. **Create embeddings:** Apply an embedding model to transform these text chunks into vector embeddings, capturing their semantic meaning.
3. **Store in a vector database:** Save the embeddings in a vector database, enabling fast retrieval based on semantic similarity.
4. **Handle user queries:** Convert the user's query into an embedding using the same model that was applied to the text chunks.
5. **Retrieve relevant data:** Search the vector database for embeddings that closely match the query’s embedding based on semantic similarity.
6. **Enhance the prompt:** Incorporate the most relevant text chunks into the LLM’s prompt to provide valuable context for generating a response.
7. **Generate a response:** The LLM leverages the augmented prompt to deliver a response that is accurate and tailored to the user’s query.

## 🎯 Approaches

RAG implementations vary in complexity, from simple document retrieval to advanced techniques integrating iterative feedback loops and domain-specific enhancements. Approaches may include:

- [Cache-Augmented Generation (CAG)](https://medium.com/@ronantech/cache-augmented-generation-cag-in-llms-a-step-by-step-tutorial-6ac35d415eec): Preloads relevant documents into a model’s context and stores the inference state (Key-Value (KV) cache).
- [Agentic RAG](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/): Also known as retrieval agents, can make decisions on retrieval processes.
- [Corrective RAG](https://arxiv.org/pdf/2401.15884.pdf) (CRAG): Methods to correct or refine the retrieved information before integration into LLM responses.
- [Retrieval-Augmented Fine-Tuning](https://techcommunity.microsoft.com/t5/ai-ai-platform-blog/raft-a-new-way-to-teach-llms-to-be-better-at-rag/ba-p/4084674) (RAFT): Techniques to fine-tune LLMs specifically for enhanced retrieval and generation tasks.
- [Self Reflective RAG](https://selfrag.github.io/): Models that dynamically adjust retrieval strategies based on model performance feedback.
- [RAG Fusion](https://arxiv.org/abs/2402.03367): Techniques combining multiple retrieval methods for improved context integration.
- [Temporal Augmented Retrieval](https://adam-rida.medium.com/temporal-augmented-retrieval-tar-dynamic-rag-ad737506dfcc) (TAR): Considering time-sensitive data in retrieval processes.
- [Plan-then-RAG](https://arxiv.org/abs/2406.12430) (PlanRAG): Strategies involving planning stages before executing RAG for complex tasks.
- [GraphRAG](https://github.com/microsoft/graphrag): A structured approach using knowledge graphs for enhanced context integration and reasoning.
- [FLARE](https://medium.com/etoai/better-rag-with-active-retrieval-augmented-generation-flare-3b66646e2a9f) - An approach that incorporates active retrieval-augmented generation to improve response quality.
- [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) - Improves retrieval by adding relevant context to document chunks before retrieval, enhancing the relevance of information retrieved from large knowledge bases.
- [GNN-RAG](https://github.com/cmavro/GNN-RAG): Graph neural retrieval for large language modeling reasoning.

## 🧰 Frameworks that Facilitate RAG

- [Haystack](https://github.com/deepset-ai/haystack): LLM orchestration framework to build customizable, production-ready LLM applications.
- [LangChain](https://python.langchain.com/docs/modules/data_connection/): An all-purpose framework for working with LLMs.
- [Semantic Kernel](https://github.com/microsoft/semantic-kernel): An SDK from Microsoft for developing Generative AI applications.
- [LlamaIndex](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/): Framework for connecting custom data sources to LLMs.
- [Dify](https://github.com/langgenius/dify): An open-source LLM app development platform.
- [Cognita](https://github.com/truefoundry/cognita): Open-source RAG framework for building modular and production ready applications.
- [Verba](https://github.com/weaviate/Verba): Open-source application for RAG out of the box.
- [Mastra](https://github.com/mastra-ai/mastra): Typescript framework for building AI applications.
- [Letta](https://github.com/letta-ai/letta): Open source framework for building stateful LLM applications.
- [Flowise](https://github.com/FlowiseAI/Flowise): Drag & drop UI to build customized LLM flows.
- [Swiftide](https://github.com/bosun-ai/swiftide): Rust framework for building modular, streaming LLM applications.
- [CocoIndex](https://github.com/cocoindex-io/cocoindex): ETL framework to index data for AI, such as RAG; with realtime incremental updates.
  
## 🛠️ Techniques

### Data cleaning

- [Data cleaning techniques](https://medium.com/intel-tech/four-data-cleaning-techniques-to-improve-large-language-model-llm-performance-77bee9003625): Pre-processing steps to refine input data and improve model performance.

### Prompting

- **Strategies**
  - [Tagging and Labeling](https://python.langchain.com/v0.1/docs/use_cases/tagging/): Adding semantic tags or labels to retrieved data to enhance relevance.
  - [Chain of Thought (CoT)](https://www.promptingguide.ai/techniques/cot): Encouraging the model to think through problems step by step before providing an answer.
  - [Chain of Verification (CoVe)](https://sourajit16-02-93.medium.com/chain-of-verification-cove-understanding-implementation-e7338c7f4cb5): Prompting the model to verify each step of its reasoning for accuracy.
  - [Self-Consistency](https://www.promptingguide.ai/techniques/consistency): Generating multiple reasoning paths and selecting the most consistent answer.
  - [Zero-Shot Prompting](https://www.promptingguide.ai/techniques/zeroshot): Designing prompts that guide the model without any examples.
  - [Few-Shot Prompting](https://python.langchain.com/docs/how_to/few_shot_examples/): Providing a few examples in the prompt to demonstrate the desired response format.
  - [Reason & Act (ReAct) prompting](https://www.promptingguide.ai/techniques/react): Combines reasoning (e.g. CoT) with acting (e.g. tool calling).
- **Caching**
  - [Prompt Caching](https://medium.com/@1kg/prompt-cache-what-is-prompt-caching-a-comprehensive-guide-e6cbae48e6a3): Optimizes LLMs by storing and reusing precomputed attention states.

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

- **Select embedding model**
  - **[MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)**: Explore [Hugging Face's](https://github.com/huggingface) benchmark for evaluating model embeddings.
  - **Custom Embeddings**: Develop tailored embeddings for specific domains or tasks to enhance model performance. Custom embeddings can capture domain-specific terminology and nuances. Techniques include fine-tuning pre-trained models on your own dataset or training embeddings from scratch using frameworks like TensorFlow or PyTorch.

### Retrieval

- **Search Methods**
  - [Vector Store Flat Index](https://weaviate.io/developers/academy/py/vector_index/flat)
    - Simple and efficient form of retrieval.
    - Content is vectorized and stored as flat content vectors.
  - [Hierarchical Index Retrieval](https://pixion.co/blog/rag-strategies-hierarchical-index-retrieval)
    - Hierarchically narrow data to different levels.
    - Executes retrievals by hierarchical order.
  - [Hypothetical Questions](https://pixion.co/blog/rag-strategies-hypothetical-questions-hyde)
    - Used to increase similarity between database chunks and queries (same with HyDE).
    - LLM is used to generate specific questions for each text chunk.
    - Converts these questions into vector embeddings.
    - During search, matches queries against this index of question vectors.
  - [Hypothetical Document Embeddings (HyDE)](https://pixion.co/blog/rag-strategies-hypothetical-questions-hyde)
    - Used to increase similarity between database chunks and queries (same with Hypothetical Questions).
    - LLM is used to generate a hypothetical response based on the query.
    - Converts this response into a vector embedding.
    - Compares the query vector with the hypothetical response vector.
  - [Small to Big Retrieval](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/retrieval-augmented-generation/small_to_big_rag/small_to_big_rag.ipynb)
    - Improves retrieval by using smaller chunks for search and larger chunks for context.
    - Smaller child chunks refers to bigger parent chunks
- **[Re-ranking](https://developer.nvidia.com/blog/enhancing-rag-pipelines-with-re-ranking/)**: Enhances search results in RAG pipelines by reordering initially retrieved documents, prioritizing those most semantically relevant to the query.

### Response quality & safety

- **[Hallucination](https://machinelearningmastery.com/rag-hallucination-detection-techniques/):** When an AI model generates incorrect or fabricated information, which can be mitigated through grounding, refined retrieval, and verification techniques.
- **[Guardrails](https://developer.ibm.com/tutorials/awb-how-to-implement-llm-guardrails-for-rag-applications/):** Mechanisms to ensure accurate, ethical, and safe responses by applying content moderation, bias mitigation, and fact-checking.
- **[Prompt Injection Prevention](https://hiddenlayer.com/innovation-hub/prompt-injection-attacks-on-llms/):**
  - **Input Validation:** Rigorously validate and sanitize all external inputs to ensure that only intended data is incorporated into the prompt.
  - **Content Separation:** Clearly distinguish between trusted, static instructions and dynamic user data using templating or placeholders.
  - **Output Monitoring:** Continuously monitor responses and logs for any anomalies that could indicate prompt manipulation, and adjust guardrails accordingly.

## 📊 Metrics

### Search metrics

These metrics are used to measure the similarity between embeddings, which is crucial for evaluating how effectively RAG systems retrieve and integrate external documents or data sources. By selecting appropriate similarity metrics, you can optimize the performance and accuracy of your RAG system. Alternatively, you may develop custom metrics tailored to your specific domain or niche to capture domain-specific nuances and improve relevance.

- **[Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)**

  - Measures the cosine of the angle between two vectors in a multi-dimensional space.
  - Highly effective for comparing text embeddings where the direction of the vectors represents semantic information.
  - Commonly used in RAG systems to measure semantic similarity between query embeddings and document embeddings.

- **[Dot Product](https://en.wikipedia.org/wiki/Dot_product)**

  - Calculates the sum of the products of corresponding entries of two sequences of numbers.
  - Equivalent to cosine similarity when vectors are normalized.
  - Simple and efficient, often used with hardware acceleration for large-scale computations.

- **[Euclidean Distance](https://en.wikipedia.org/wiki/Euclidean_distance)**

  - Computes the straight-line distance between two points in Euclidean space.
  - Can be used with embeddings but may lose effectiveness in high-dimensional spaces due to the "[curse of dimensionality](https://stats.stackexchange.com/questions/99171/why-is-euclidean-distance-not-a-good-metric-in-high-dimensions)."
  - Often used in clustering algorithms like K-means after dimensionality reduction.

- **[Jaccard Similarity](https://en.wikipedia.org/wiki/Jaccard_index)**
  - Measures the similarity between two finite sets as the size of the intersection divided by the size of the union of the sets.
  - Useful when comparing sets of tokens, such as in bag-of-words models or n-gram comparisons.
  - Less applicable to continuous embeddings produced by LLMs.

> **Note:** Cosine Similarity and Dot Product are generally seen as the most effective metrics for measuring similarity between high-dimensional embeddings.

### Response Evaluation Metrics

Response evaluation in RAG solutions involves assessing the quality of language model outputs using diverse metrics. Here are structured approaches to evaluating these responses:

- **Automated Benchmarking**

  - **[BLEU](https://en.wikipedia.org/wiki/BLEU):** Evaluates the overlap of n-grams between machine-generated and reference outputs, providing insight into precision.
  - **[ROUGE](<https://en.wikipedia.org/wiki/ROUGE_(metric)>):** Measures recall by comparing n-grams, skip-bigrams, or longest common subsequence with reference outputs.
  - **[METEOR](https://en.wikipedia.org/wiki/METEOR):** Focuses on exact matches, stemming, synonyms, and alignment for machine translation.

- **Human Evaluation**
  Involves human judges assessing responses for:

  - **Relevance:** Alignment with user queries.
  - **Fluency:** Grammatical and stylistic quality.
  - **Factual Accuracy:** Verifying claims against authoritative sources.
  - **Coherence:** Logical consistency within responses.

- **Model Evaluation**
  Leverages pre-trained evaluators to benchmark outputs against diverse criteria:

  - **[TuringBench](https://turingbench.ist.psu.edu/):** Offers comprehensive evaluations across language benchmarks.
  - **[Hugging Face Evaluate](https://huggingface.co/docs/evaluate/en/index):** Calculates alignment with human preferences.

- **Key Dimensions for Evaluation**
  - **Groundedness:** Assesses if responses are based entirely on provided context. Low groundedness may indicate reliance on hallucinated or irrelevant information.
  - **Completeness:** Measures if the response answers all aspects of a query.
  - **Approaches:** AI-assisted retrieval scoring and prompt-based intent verification.
  - **Utilization:** Evaluates the extent to which retrieved data contributes to the response.
  - **Analysis:** Use LLMs to check the inclusion of retrieved chunks in responses.

#### Tools

These tools can assist in evaluating the performance of your RAG system, from tracking user feedback to logging query interactions and comparing multiple evaluation metrics over time.

- **[LangFuse](https://github.com/langfuse/langfuse)**: Open-source tool for tracking LLM metrics, observability, and prompt management.
- **[Ragas](https://docs.ragas.io/en/stable/)**: Framework that helps evaluate RAG pipelines.
- **[LangSmith](https://docs.smith.langchain.com/)**: A platform for building production-grade LLM applications, allows you to closely monitor and evaluate your application.
- **[Hugging Face Evaluate](https://github.com/huggingface/evaluate)**: Tool for computing metrics like BLEU and ROUGE to assess text quality.
- **[Weights & Biases](https://wandb.ai/wandb-japan/rag-hands-on/reports/Step-for-developing-and-evaluating-RAG-application-with-W-B--Vmlldzo1NzU4OTAx)**: Tracks experiments, logs metrics, and visualizes performance.

## 💾 Databases

The list below features several database systems suitable for Retrieval Augmented Generation (RAG) applications. They cover a range of RAG use cases, aiding in the efficient storage and retrieval of vectors to generate responses or recommendations.

### Benchmarks

- [Picking a vector database](https://benchmark.vectorview.ai/vectordbs.html)

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

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Danielskry/Awesome-RAG&type=Date)](https://www.star-history.com/#Danielskry/Awesome-RAG&Date)

---

This list continues to evolve. Contributions are welcome to make this resource more comprehensive 🙌
