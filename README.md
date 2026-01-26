# 😎 Awesome Retrieval Augmented Generation (RAG) [![Awesome](https://awesome.re/badge-flat.svg)](https://awesome.re)

This repository contains a curated [Awesome List](https://github.com/sindresorhus/awesome) and comprehensive technical information on Retrieval-Augmented Generation (RAG) applications in Generative AI.

## Overview

**Retrieval-Augmented Generation (RAG)** is a sophisticated technique in Generative AI that enhances Large Language Models (LLMs) by dynamically retrieving and incorporating relevant context from external knowledge sources during the generation process. Unlike traditional LLMs that rely solely on pre-trained knowledge, RAG systems enable models to access up-to-date, domain-specific, or proprietary information, significantly improving accuracy, reducing hallucinations, and enabling real-time knowledge integration.

### Key Benefits

- **Reduced Hallucinations**: Grounds responses in retrieved factual information
- **Domain Adaptation**: Enables LLMs to work with specialized knowledge without fine-tuning
- **Real-time Updates**: Incorporates latest information without model retraining
- **Cost Efficiency**: More economical than fine-tuning for domain-specific tasks
- **Transparency**: Provides source attribution for generated content
- **Privacy & Security**: Keeps sensitive data in private knowledge bases

## Content

- [ℹ️ General Information on RAG](#ℹ%EF%B8%8F-general-information-on-rag)
- [🏗️ Architecture Patterns](#%EF%B8%8F-architecture-patterns)
- [🎯 Advanced Approaches](#-advanced-approaches)
- [🧰 Frameworks that Facilitate RAG](#-frameworks-that-facilitate-rag)
- [🐍 Python Ecosystem for RAG](#-python-ecosystem-for-rag)
- [🛠️ Techniques](#-techniques)
- [📊 Metrics & Evaluation](#-metrics--evaluation)
- [💾 Databases](#-databases)
- [🔌 Platform-Specific RAG Implementations](#-platform-specific-rag-implementations)
- [🚀 Production Considerations](#-production-considerations)
- [💡 Best Practices](#-best-practices)

## ℹ️ General Information on RAG

RAG addresses a fundamental limitation of LLMs: their static knowledge cutoff and inability to access external information. Traditional RAG implementations employ a retrieval pipeline that enriches LLM prompts with contextually relevant documents from a knowledge base. For example, when querying about renovation materials for a specific house, the LLM may have general renovation knowledge but lacks details about that particular property. An RAG system can retrieve relevant documents (e.g., blueprints, material specifications, local building codes) to provide accurate, context-aware responses.

### Core RAG Pipeline

A standard RAG implementation follows these critical steps:

1. **Document Ingestion & Preprocessing**: 
   - Collect and normalize documents from various sources (PDFs, databases, APIs, web content)
   - Apply data cleaning, deduplication, and format standardization

2. **Chunking Strategy**: 
   - Divide documents into semantically meaningful chunks
   - Balance chunk size to preserve context while enabling precise retrieval
   - Consider document structure, domain-specific boundaries, and overlap strategies

3. **Embedding Generation**: 
   - Transform text chunks into high-dimensional vector embeddings using embedding models
   - Select models optimized for your domain (general-purpose vs. domain-specific)
   - Ensure embedding consistency across indexing and querying phases

4. **Vector Storage & Indexing**: 
   - Store embeddings in a vector database optimized for similarity search
   - Build efficient indexes (HNSW, IVF, etc.) for fast retrieval at scale
   - Maintain metadata for filtering, hybrid search, and result interpretation

5. **Query Processing**: 
   - Convert user queries into embeddings using the same model
   - Apply query expansion, reformulation, or decomposition for complex queries
   - Support multi-turn conversations with context preservation

6. **Retrieval & Ranking**: 
   - Perform similarity search to find relevant chunks
   - Apply re-ranking models to improve precision
   - Implement hybrid search combining semantic and keyword-based retrieval

7. **Context Augmentation**: 
   - Construct prompts with retrieved context and clear instructions
   - Manage context window limits and prioritize most relevant information
   - Include source attribution for transparency

8. **Generation & Post-processing**: 
   - Generate responses using the augmented prompt
   - Apply safety checks, fact verification, and quality filters
   - Format output with citations and confidence indicators

### Implementation Resources

#### Python Tutorials & Examples

- Complete basic [RAG implementation in Python](https://github.com/Danielskry/LangChain-Chroma-RAG-demo-2024): Full-stack RAG example with LangChain and Chroma
- [LangChain RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/): Comprehensive guide to building RAG applications
- [LlamaIndex RAG Tutorial](https://docs.llamaindex.ai/en/stable/getting_started/starter_example/): Getting started with LlamaIndex for RAG
- [Haystack RAG Pipeline](https://docs.haystack.deepset.ai/docs/retrieval-augmented-generation): Building RAG pipelines with Haystack

#### Production & Best Practices

- [Production RAG patterns and best practices](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/): Production-ready RAG optimization strategies
- [LangChain Production Guide](https://python.langchain.com/docs/production/): Deploying LangChain applications to production
- [Python Async Best Practices](https://docs.python.org/3/library/asyncio-dev.html): Writing efficient async Python code for AI applications

## 🏗️ Architecture Patterns

RAG systems can be architected using various patterns depending on requirements:

- **Naive RAG**: Basic retrieve-then-generate pipeline without optimization
- **Advanced RAG**: Incorporates query rewriting, re-ranking, and context compression
- **Modular RAG**: Composable components for retrieval, ranking, and generation
- **Agentic RAG**: LLM-driven agents that make retrieval decisions dynamically
- **Self-RAG**: Models that self-reflect on retrieval quality and adjust strategies
- **Graph RAG**: Leverages knowledge graphs for structured information retrieval

## 🎯 Advanced Approaches

RAG implementations vary in complexity, from simple document retrieval to advanced techniques integrating iterative feedback loops, multi-agent systems, and domain-specific enhancements. Modern approaches include:

- [Vision-RAG](https://www.youtube.com/watch?v=npkp4mSweEg): Embeds entire pages as images, allowing vision models to handle reasoning directly without parsing text-RAG.
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
- [GNN-RAG](https://github.com/cmavro/GNN-RAG): Graph neural retrieval for large language modeling reasoning.
- [Multimodal RAG](https://developer.nvidia.com/blog/an-easy-introduction-to-multimodal-retrieval-augmented-generation/): Extends RAG to handle multiple modalities such as text, images, and audio.
- [VideoRAG](https://arxiv.org/abs/2501.05874): Extends RAG to videos using Large Video Language Models (LVLMs) to retrieve and integrate visual and textual content for multimodal generation.
- [REFRAG](https://arxiv.org/pdf/2509.01092): Optimizes RAG decoding by compressing retrieved context into embeddings before generation, reducing latency while maintaining output quality.
- [InstructRAG](https://github.com/weizhepei/InstructRAG): Instructing RAG via Self-Synthesized Rationales 

## 🧰 Frameworks that Facilitate RAG

- [Haystack](https://github.com/deepset-ai/haystack): LLM orchestration framework to build customizable, production-ready LLM applications.
- [LangChain](https://python.langchain.com/docs/modules/data_connection/): An all-purpose framework for working with LLMs.
- [Semantic Kernel](https://github.com/microsoft/semantic-kernel): An SDK from Microsoft for developing Generative AI applications.
- [LlamaIndex](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/): Framework for connecting custom data sources to LLMs. Features [official Supabase integration](https://supabase.com/docs/guides/ai/integrations/llamaindex) for seamless RAG development.
- [Dify](https://github.com/langgenius/dify): An open-source LLM app development platform.
- [Cognita](https://github.com/truefoundry/cognita): Open-source RAG framework for building modular and production ready applications.
- [Verba](https://github.com/weaviate/Verba): Open-source application for RAG out of the box.
- [Mastra](https://github.com/mastra-ai/mastra): Typescript framework for building AI applications.
- [Letta](https://github.com/letta-ai/letta): Open source framework for building stateful LLM applications.
- [Flowise](https://github.com/FlowiseAI/Flowise): Drag & drop UI to build customized LLM flows.
- [Swiftide](https://github.com/bosun-ai/swiftide): Rust framework for building modular, streaming LLM applications.
- [CocoIndex](https://github.com/cocoindex-io/cocoindex): ETL framework to index data for AI, such as RAG; with realtime incremental updates.
- [Pathway](https://github.com/pathwaycom/pathway/): Performant open-source Python ETL framework with Rust runtime, supporting 300+ data sources.
- [Pathway AI Pipelines](https://github.com/pathwaycom/llm-app/): A production-ready RAG framework supporting real-time indexing, retrieval, and change tracking across diverse data sources.
- [LiteLLM](https://docs.litellm.ai/observability/supabase_integration): Unified interface for multiple LLM providers (OpenAI, Anthropic, Hugging Face, Replicate) with [Supabase integration](https://docs.litellm.ai/observability/supabase_integration) for logging, monitoring, and cost tracking.

## 🐍 Python Ecosystem for RAG

Python is the most mature ecosystem for RAG today, with extensive support for
LLMs, embeddings, vector databases, evaluation, and production tooling.

👉 See the full guide: [Python Ecosystem for RAG](docs/python-ecosystem.md)

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
- **Structuring**
  -  [Token-Oriented Object Notation](https://github.com/toon-format/toon): A compact, deterministic JSON format for LLM prompts.

### Chunking

Chunking strategy is one of the most critical decisions in RAG system design, directly impacting retrieval precision and context quality. The optimal approach depends on document types, domain characteristics, and query patterns.

- **[Fixed-Size Chunking](https://medium.com/@anuragmishra_27746/five-levels-of-chunking-strategies-in-rag-notes-from-gregs-video-7b735895694d)**
  - **Use Case**: Simple, uniform documents where structure is less important
  - **Characteristics**: Divides text into consistent-sized segments (typically 256-512 tokens) with configurable overlap (10-20%)
  - **Pros**: Simple to implement, predictable chunk sizes, efficient processing
  - **Cons**: May split sentences/paragraphs, loses document structure, can fragment semantic units
  - **Implementation**: [CharacterTextSplitter](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/character_text_splitter/) (LangChain), [SentenceSplitter](https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/sentence_splitter/) (LlamaIndex)

- **[Recursive Chunking](https://medium.com/@AbhiramiVS/chunking-methods-all-to-know-about-it-65c10aa7b24e)**
  - **Use Case**: Documents with hierarchical structure (markdown, HTML, code)
  - **Characteristics**: Recursively splits by separators (paragraphs → sentences → words) until desired chunk size
  - **Pros**: Preserves natural boundaries, respects document hierarchy, better semantic coherence
  - **Cons**: More complex, variable chunk sizes, requires careful separator configuration
  - **Implementation**: [RecursiveCharacterTextSplitter](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/) (LangChain)

- **[Document-Based Chunking](https://medium.com/@david.richards.tech/document-chunking-for-rag-ai-applications-04363d48fbf7)**
  - **Use Case**: Structured documents with clear sections (markdown headers, PDF sections, database records)
  - **Characteristics**: Segments based on document metadata, formatting cues, or structural elements
  - **Pros**: Maintains document structure, preserves context, enables metadata-rich retrieval
  - **Cons**: Requires structured input, may create very large or very small chunks
  - **Implementation**: [MarkdownHeaderTextSplitter](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/markdown_header_metadata/) (LangChain)
  - **Multimodal**: Handle images and text with models like [OpenCLIP](https://github.com/mlfoundations/open_clip)

- **[Semantic Chunking](https://www.youtube.com/watch?v=8OJC21T2SL4&t=1933s)**
  - **Use Case**: Documents where semantic coherence is critical (narratives, technical documentation)
  - **Characteristics**: Uses embedding similarity to identify natural semantic boundaries
  - **Pros**: Preserves semantic units, adapts to content, improves retrieval relevance
  - **Cons**: Computationally expensive, requires embedding model, less predictable chunk sizes
  - **Best For**: High-quality retrieval where context preservation is paramount

- **[Agentic Chunking](https://youtu.be/8OJC21T2SL4?si=8VnYaGUaBmtZhCsg&t=2882)**
  - **Use Case**: Complex documents requiring intelligent segmentation decisions
  - **Characteristics**: Uses LLMs to analyze content and determine optimal chunk boundaries
  - **Pros**: Highly adaptive, understands context, can apply domain knowledge
  - **Cons**: High cost, slower processing, requires LLM API access
  - **Best For**: Specialized domains where standard chunking fails

**Chunking Best Practices:**
- **Overlap Strategy**: Use 10-20% overlap to maintain context across boundaries
- **Size Optimization**: Balance chunk size (larger = more context, smaller = better precision)
- **Metadata Preservation**: Retain document structure, headers, and formatting in chunk metadata
- **Multi-Granularity**: Consider hierarchical approaches (small chunks for retrieval, larger for context)

### Embeddings

Embeddings are the foundation of semantic search in RAG systems. The choice of embedding model significantly impacts retrieval quality.

- **Model Selection**
  - **[MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)**: Comprehensive benchmark for evaluating embedding models across multiple tasks and languages. Consider models that perform well on tasks relevant to your use case (retrieval, clustering, classification).
  - **Model Characteristics**: Evaluate models based on:
    - **Dimensions**: Higher dimensions (768-1024) generally offer better quality but increase storage and compute costs
    - **Context Length**: Ensure models support your document chunk sizes
    - **Multilingual Support**: Required for international applications
    - **Domain Specialization**: General-purpose vs. domain-specific (e.g., scientific, legal, medical)
  
- **Custom Embeddings**
  - **Fine-tuning**: Adapt pre-trained models to your domain using contrastive learning, triplet loss, or supervised fine-tuning
  - **Training from Scratch**: For highly specialized domains with sufficient labeled data
  - **Multi-Modal Embeddings**: For applications requiring text, image, or audio understanding (e.g., CLIP, ImageBind)
  - **Ensemble Methods**: Combine multiple embedding models for improved robustness

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
  - [Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval)
    - Enhances RAG retrieval accuracy by preserving document context that is typically lost during chunking.
    - Each text chunk is enriched with a short, model-generated summary before embedding and indexing, resulting in Contextual Embeddings and Contextual BM25.
    - This combined approach improves both semantic and lexical matching, reducing retrieval failure rates when paired with reranking.
  - [Adaptive Retrieval](https://arxiv.org/abs/2403.14403)
    - Dynamically decide when and how much to retrieve during generation.
  - [Query Reformulation and Expansion](https://haystack.deepset.ai/cookbook/query-expansion)
    - Automatically rewrites or expands the query before retrieval to boost recall.
    - Useful for long or ambiguous user queries.
- **[Re-ranking](https://developer.nvidia.com/blog/enhancing-rag-pipelines-with-re-ranking/)**: Enhances search results in RAG pipelines by reordering initially retrieved documents, prioritizing those most semantically relevant to the query.

### Response Quality & Safety

Ensuring high-quality, safe, and reliable responses is critical for production RAG systems.

- **Hallucination Mitigation**
  - **[Detection Techniques](https://machinelearningmastery.com/rag-hallucination-detection-techniques/)**: Implement methods to identify when models generate unsupported information
  - **Grounding Verification**: Cross-reference generated claims with retrieved context
  - **Confidence Scoring**: Assign confidence scores to generated responses based on source quality
  - **Source Attribution**: Require citations for all factual claims
  - **Retrieval Quality**: Improve retrieval precision to reduce hallucination risk

- **Guardrails & Safety**
  - **[Implementation Guide](https://developer.ibm.com/tutorials/awb-how-to-implement-llm-guardrails-for-rag-applications/)**: Comprehensive approach to implementing safety mechanisms
  - **Content Moderation**: Filter harmful, biased, or inappropriate content at input and output stages
  - **Bias Mitigation**: Detect and mitigate biases in retrieved content and generated responses
  - **Fact-Checking**: Verify claims against authoritative sources or knowledge bases
  - **Toxicity Detection**: Use classifiers to identify and filter toxic content

- **Prompt Injection Prevention**
  - **[Security Guide](https://hiddenlayer.com/innovation-hub/prompt-injection-attacks-on-llms/)**: Understanding and preventing prompt injection attacks
  - **Input Validation**: Rigorously validate and sanitize all external inputs using whitelisting, length limits, and pattern matching
  - **Content Separation**: Use clear delimiters, templating systems, and role-based prompts to separate instructions from user data
  - **Output Monitoring**: Continuously monitor responses for anomalies, unexpected behaviors, or security violations
  - **Rate Limiting**: Implement rate limits and abuse detection to prevent systematic attacks
  - **Sandboxing**: Isolate LLM execution environments to limit potential damage from successful injections
  - **Platform-Specific Security**: For Supabase implementations, leverage [Row Level Security (RLS)](https://supabase.com/docs/guides/auth/row-level-security) policies to restrict access to vector data and implement [Edge Function authentication](https://supabase.com/docs/guides/functions/auth) for secure AI model access

## 📊 Metrics & Evaluation

### Similarity Metrics for Embeddings

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

Vector databases are critical components of RAG systems, providing efficient storage and similarity search capabilities for embeddings. The selection of an appropriate database depends on factors such as scale, latency requirements, deployment model (cloud vs. on-premises), and feature needs (hybrid search, filtering, etc.). The list below features database systems suitable for RAG applications:

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
- [Supabase](https://supabase.com/): Open-source Firebase alternative built on PostgreSQL with native pgvector support, real-time subscriptions, and integrated AI/LLM capabilities. Features include:
  - **Vector Search**: Native pgvector integration for efficient similarity search
  - **Real-time RAG**: Real-time document updates and query subscriptions
  - **Edge Functions**: Serverless functions for embedding generation and LLM integration
  - **Row Level Security**: Fine-grained access control for secure RAG applications
  - **LlamaIndex Integration**: [Official integration](https://supabase.com/docs/guides/ai/integrations/llamaindex) for seamless RAG pipeline development
  - [Supabase AI Integrations](https://supabase.com/features/ai-integrations): Comprehensive AI platform connectivity

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

## 🚀 Production Considerations

Building production-grade RAG systems requires addressing several critical aspects beyond the core retrieval and generation pipeline:

### Scalability & Performance

- **Indexing Throughput**: Design pipelines to handle high-volume document ingestion with incremental updates
- **Query Latency**: Optimize retrieval speed through efficient indexing (HNSW, IVF), caching strategies, and parallel processing
- **Concurrent Requests**: Implement connection pooling, request queuing, and load balancing for high-traffic scenarios
- **Resource Management**: Monitor GPU/CPU utilization, memory consumption, and database connection pools

### Reliability & Monitoring

- **Observability**: Implement comprehensive logging, tracing, and metrics collection (latency, throughput, error rates)
- **Health Checks**: Monitor embedding service availability, vector database connectivity, and LLM API status
- **Error Handling**: Implement retry logic, circuit breakers, and graceful degradation strategies
- **A/B Testing**: Compare different retrieval strategies, chunking methods, and prompt templates

### Data Management

- **Incremental Updates**: Support real-time or near-real-time document indexing without full re-indexing
- **Version Control**: Track document versions, embedding model versions, and prompt templates
- **Data Quality**: Implement validation pipelines to detect corrupted embeddings, missing metadata, or stale content
- **Backup & Recovery**: Regular backups of vector indexes and metadata stores

### Security & Compliance

- **Access Control**: Implement authentication, authorization, and audit logging
- **Data Privacy**: Encrypt data at rest and in transit, support data residency requirements
- **Content Filtering**: Apply content moderation, PII detection, and compliance checks
- **Rate Limiting**: Protect against abuse and ensure fair resource allocation

### Cost Optimization

- **Embedding Caching**: Cache frequently accessed embeddings to reduce API costs
- **Selective Retrieval**: Use query routing to avoid unnecessary retrieval operations
- **Model Selection**: Balance cost and performance when choosing embedding and LLM models
- **Resource Right-sizing**: Optimize infrastructure based on actual usage patterns
- **Platform-Specific Optimization**: 
  - For Supabase: Use [LiteLLM integration](https://docs.litellm.ai/observability/supabase_integration) for cost tracking and multi-provider fallback strategies
  - Leverage Supabase Edge Functions for efficient serverless execution and automatic scaling
  - Utilize connection pooling to reduce database connection overhead

## 🔌 Platform-Specific RAG Implementations

### Supabase RAG Architecture

Supabase provides a comprehensive platform for building production-ready RAG systems with integrated AI/LLM capabilities:

#### Core Components

- **PostgreSQL + pgvector**: Native vector similarity search with HNSW indexing for fast retrieval
- **Real-time Subscriptions**: Live updates to document embeddings and query results
- **Edge Functions**: Serverless execution for:
  - Embedding generation using OpenAI, Cohere, or custom models
  - LLM integration (OpenAI, Anthropic, Hugging Face, Replicate)
  - Query processing and context augmentation
  - Response generation and post-processing

#### Integration Patterns

- **LlamaIndex + Supabase**: 
  - [Official integration guide](https://supabase.com/docs/guides/ai/integrations/llamaindex)
  - Seamless data ingestion, chunking, and retrieval workflows
  - Built-in support for document loaders and vector stores

- **LiteLLM + Supabase**:
  - [Unified LLM provider interface](https://docs.litellm.ai/observability/supabase_integration)
  - Request logging and cost monitoring
  - Multi-provider fallback strategies

#### Security & Access Control

- **Row Level Security (RLS)**: Fine-grained access control for vector data
- **API Authentication**: Secure access to embeddings and LLM endpoints
- **Data Isolation**: Tenant-specific vector namespaces for multi-tenant applications

#### Production Features

- **Automatic Backups**: Built-in PostgreSQL backup and point-in-time recovery
- **Connection Pooling**: Optimized database connections for high-concurrency RAG queries
- **Monitoring & Observability**: Integrated logging and metrics for RAG pipeline performance
- **Scalability**: Horizontal scaling with read replicas for vector search workloads

#### Example Use Cases

- **Document Q&A Systems**: Real-time question answering over private knowledge bases
- **AI-Powered Search**: Semantic search with hybrid keyword + vector retrieval
- **Chatbots with Memory**: Conversational AI with persistent context storage
- **Content Recommendation**: Similarity-based content discovery and personalization

#### Getting Started

- [Supabase AI Integrations Documentation](https://supabase.com/features/ai-integrations)
- [Supabase Vector Search Guide](https://supabase.com/docs/guides/database/extensions/pgvector)
- [Building AI Agents with Supabase](https://www.youtube.com/watch?v=8GH-afNDebI): Complete tutorial

## 💡 Best Practices

### Chunking Strategy

- **Domain-Aware Chunking**: Use semantic or document-structure-based chunking over fixed-size for better context preservation
- **Overlap Management**: Include strategic overlap (10-20%) to maintain context across boundaries
- **Metadata Preservation**: Retain document structure, headers, and formatting cues in chunk metadata
- **Multi-Granularity**: Consider hierarchical chunking (small chunks for retrieval, larger chunks for context)

### Embedding Selection

- **Model Evaluation**: Use MTEB leaderboard and domain-specific benchmarks to select appropriate models
- **Dimension Optimization**: Balance embedding dimensions (higher = better quality, lower = faster retrieval)
- **Domain Fine-tuning**: Fine-tune embeddings on domain-specific data when possible
- **Consistency**: Ensure the same embedding model is used for indexing and querying

### Retrieval Optimization

- **Hybrid Search**: Combine semantic (vector) and lexical (BM25/keyword) search for improved recall
- **Re-ranking**: Apply cross-encoders or learned-to-rank models to improve precision
- **Query Understanding**: Implement query classification, intent detection, and query expansion
- **Result Diversification**: Avoid redundant results by implementing diversity constraints

### Prompt Engineering

- **Clear Instructions**: Provide explicit instructions on how to use retrieved context
- **Source Attribution**: Request citations and require grounding in provided context
- **Few-Shot Examples**: Include examples demonstrating desired response format and quality
- **Context Compression**: Use techniques like summarization or extraction when context exceeds limits

### Evaluation Framework

- **Multi-Dimensional Metrics**: Evaluate relevance, accuracy, completeness, and groundedness
- **Human-in-the-Loop**: Incorporate human feedback for continuous improvement
- **Synthetic Evaluation**: Generate test queries and expected outputs for automated testing
- **Production Monitoring**: Track user satisfaction, query patterns, and failure modes

### Iterative Improvement

- **Feedback Loops**: Collect user feedback, query logs, and performance metrics
- **Experimentation**: Systematically test improvements (chunking, retrieval, prompts) with controlled experiments
- **Model Updates**: Plan for embedding model upgrades and migration strategies
- **Documentation**: Maintain clear documentation of architecture, decisions, and operational procedures

### Python-Specific Best Practices

- **Virtual Environments**: Use `venv` or `conda` to manage dependencies and Python versions
- **Dependency Management**: Use `requirements.txt` or `pyproject.toml` with version pinning
- **Code Quality**: Use `black` for formatting, `flake8` or `ruff` for linting, and `mypy` for type checking
- **Testing**: Write unit tests with `pytest`, integration tests for RAG pipelines, and use mocking for external APIs
- **Logging**: Use structured logging with `structlog` or `loguru` for better observability
- **Error Handling**: Implement comprehensive error handling with custom exceptions and retry logic
- **Performance**: Profile code with `cProfile` or `py-spy`, optimize bottlenecks, and use async/await for I/O-bound operations
- **Security**: Validate inputs with Pydantic, sanitize user queries, and secure API keys using environment variables

---

This list continues to evolve. Contributions are welcome to make this resource more comprehensive 🙌
