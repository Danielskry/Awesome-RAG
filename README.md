# Awesome RAG
A curated list of retrieval-augmented generation (RAG) in large language models.  
*Suggest and discuss possible enhancements on the [Potential Additions](https://github.com/coree/awesome-rag/discussions/1) page.*


![cc](https://github.com/coree/awesome-rag/assets/5042747/de9c3103-3959-4942-9a52-02156c4bf3a4)



**Table of Content:**
- [Awesome RAG](#awesome-rag)
  - [Papers](#papers)
    - [Survey](#survey)
    - [General](#general)
  - [Resources](#resources)
    - [Lectures, Talks, Tutorials](#lectures-talks-tutorials)
    - [Workshops](#workshops)
  - [Tools](#tools)
  - [Other Collections](#other-collections)

## Papers

<!-- Paper Template
- **Title**  
  [`Paper`](Link) [`Code`](Link) [`Blog`](Link) `[Tag]` ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F[SEMANTIC_SCHOLAR_PAPER_ID]%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)   
-->

<!-- Citation Count Badge
![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F[SEMANTIC_SCHOLAR_PAPER_ID]%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)
-->
### Survey

**2024**

- **Retrieval-Augmented Generation for Large Language Models: A Survey**  
[`Paper`](https://arxiv.org/pdf/2312.10997.pdf)  [`Code`](https://github.com/Tongji-KGLLM/RAG-Survey) `arXiv` ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F46f9f7b8f88f72e12cbdb21e3311f995eb6e65c5%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)

- **Retrieval-Augmented Generation for Natural Language
Processing: A Survey**  
[`Paper`](https://arxiv.org/abs/2407.13193) `arXiv` ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fd4a5c2ab2b459426869e1a3ab1550897b005303e%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)

**2023**

- **Benchmarking Large Language Models in Retrieval-Augmented Generation**  
[`Paper`](https://arxiv.org/abs/2309.01431) `arXiv` ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F28e2ecb4183ebc0eec504b12dddc677f8aef8745%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)

**2022**

- **A Survey on Retrieval-Augmented Text Generation**  
[`Paper`](https://arxiv.org/abs/2202.01110) `arXiv` ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fe6770e3f5e74210c6863aaeed527ac4c1da419d7%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)

### General

**2024**

- **Learning to Retrieve In-Context Examples for Large Language Models**  
[`Paper`](https://arxiv.org/abs/2307.07164) [`Code`](https://github.com/microsoft/LMOps/tree/main/llm_retriever) `EACL` ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fae22f7c57916562e2729a1a7f34298e4220b77a7%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)

- **Reliable, Adaptable, and Attributable Language Models with Retrieval**  
  [`Paper`](https://arxiv.org/abs/2403.03187) `arXiv` ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fcfa85c8db829dbd2384ea7f130f462e7e7f1f630%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)

**2023**
- **Active Retrieval Augmented Generation**  
[`Paper`](https://arxiv.org/abs/2305.06983) [`Code`](https://github.com/jzbjyb/FLARE) `EMNLP` `Architecture` ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F88884b8806262a4095036041e3567d450dba39f7%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)

- ⭐ **REPLUG: Retrieval-Augmented Black-Box Language Models**  
[`Paper`](https://arxiv.org/abs/2301.12652) `arXiv` `Architecture` ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F07b14c24833400b79978b0a5f084803337e30a15%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)

- **Shall We Pretrain Autoregressive Language Models with Retrieval? A Comprehensive Study**  
[`Paper`](https://arxiv.org/abs/2304.06762) [`Code`](https://github.com/NVIDIA/Megatron-LM/tree/InstructRetro/tools/retro) `EMNLP` ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fb63e97330154acece935ffa6901e3f36518e5703%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)

- **InstructRetro: Instruction Tuning post Retrieval-Augmented Pretraining**  
[`Paper`](https://arxiv.org/abs/2310.07713) [`Code`](https://github.com/NVIDIA/Megatron-LM/tree/InstructRetro/tools/retro) `arXiv` ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F675c87c9fed17b6dc1d9734606e12c9d0c46c573%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)

- **Retrieve Anything To Augment Large Language Models**  
[`Paper`](https://arxiv.org/abs/2310.07554) [`Code`](https://github.com/FlagOpen/FlagEmbedding) `arXiv` ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F232e07b0ef0148c5325fda96eb9057c7a6db2ec2%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)

- **Reimagining Retrieval Augmented Language Models for Answering Queries**  
[`Paper`](https://arxiv.org/abs/2306.01061)  `ACL` ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fc398de8d4a18ec49b8f2eaaf3b0473186b99e1e1%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)

- **In-Context Retrieval-Augmented Language Models**   
[`Paper`](https://arxiv.org/abs/2302.00083) [`Code`](https://github.com/AI21Labs/in-context-ralm) `TACL`  `Architecture` ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F465471bb5bf1a945549d6291c2d23367966b4957%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)

- **Query Rewriting for Retrieval-Augmented Large Language Models**  
[`Paper`](https://arxiv.org/abs/2305.14283) [`Code`](https://github.com/xbmxb/RAG-query-rewriting) `EMNLP` ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Ff743287be3ced6757de7ecb26d03815b22cd737b%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)

- **Pre-computed memory or on-the-fly encoding? A hybrid approach to retrieval augmentation makes the most of your compute**  
[`Paper`](https://arxiv.org/abs/2301.10448)  `PMLR` ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F5db8c4cc8742f410d6c40a3f23eeb4739d10d0fe%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)

- **Universal Information Extraction with Meta-Pretrained Self-Retrieval**   
[`Paper`](https://aclanthology.org/2023.findings-acl.251/) [`Code`](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/metaretriever) `ACL` ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fac902460f17c3dedf40241917a86f48c4e30dd30%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)

- **RAVEN: In-Context Learning with Retrieval Augmented Encoder-Decoder Language Models**  
[`Paper`](https://arxiv.org/abs/2308.07922) `arXiv` ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F76c8e90dfd0f1e78e6a94d702a5b14b3e7206003%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)

- **Unlimiformer: Long-Range Transformers with Unlimited Length Input**  
[`Paper`](https://arxiv.org/abs/2305.01625) [`Code`](https://github.com/abertsch72/unlimiformer) `NeurIPS` `Architecture` ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fdbc368bc8b49347dd27679894524fa62f88492c9%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)

- **Nonparametric Masked Language Modeling**  
[`Paper`](https://arxiv.org/abs/2212.01349) [`Code`](https://github.com/facebookresearch/NPM) `ACL` `Training` ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F9492ee1435e183cb62b65d8d7f39be0dfd17377a%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)

**2022**

- **Improving language models by retrieving from trillions of tokens**   
[`Paper`](https://arxiv.org/abs/2112.04426) [`Blog`](https://deepmind.google/discover/blog/improving-language-models-by-retrieving-from-trillions-of-tokens/) `PMLR` `Architecture` ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F002c256d30d6be4b23d365a8de8ae0e67e4c9641%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)

- **Recitation-Augmented Language Models**  
[`Paper`](https://arxiv.org/abs/2210.01296) [`Code`](https://github.com/Edward-Sun/RECITE) `ICLR` ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fed99a2572fb5f4240aa6068e3bf274832e831306%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)

- ⭐ **Atlas: Few-shot Learning with Retrieval Augmented Language Models**  
[`Paper`](https://arxiv.org/abs/2208.03299) [`Code`](https://github.com/facebookresearch/atlas) [`Blog`](https://research.facebook.com/blog/2023/1/atlas-few-shot-learning-with-retrieval-augmented-language-models/) `Training` `JMLR` ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F398e4061dde8f5c80606869cebfa2031de7b5b74%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)

- **You can't pick your neighbors, or can you? When and how to rely on retrieval in the kNN-LM**  
[`Paper`](https://arxiv.org/abs/2210.15859) `ACL` `Architecture` ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fe832d22ca901346f50e8367afb99bd2bf6e29421%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)

- **Neuro-Symbolic Language Modeling with Automaton-augmented Retrieval**  
[`Paper`](https://arxiv.org/abs/2201.12431) [`Code`](https://github.com/neulab/retomaton) `ICML` `Architecture` ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Ff92535edac9d1c735feabdb4d94c1157f12d899c%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)

- **Memorizing Transformers**  
[`Paper`](https://arxiv.org/abs/2203.08913) [`Code`](https://github.com/princeton-nlp/TRIME) `ICLR` `Architecture` ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F0e802c0739771acf70e60d59c2df51cd7e8c50c0%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)

- **Training Language Models with Memory Augmentation**  
[`Paper`](https://arxiv.org/abs/2205.12674) `EMNLP` `Training` ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fda1d6445b6b64ce9eb4587ba8abbdc490f648ec1%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)

- **Unsupervised Dense Information Retrieval with Contrastive Learning**  
[`Paper`](https://arxiv.org/abs/2112.09118) [`Code`](https://github.com/facebookresearch/contriever) `arXiv` `Training` ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F4f4a409f701f7552d45c46a5b0fea69dca6f8e84%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)

- **Teaching language models to support answers with verified quotes**  
[`Paper`](https://arxiv.org/abs/2203.11147) `arXiv` `Application` ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F8666f9f379389a5dff31e72fb0f992a37763ba41%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)

- **kNN-Prompt: Nearest Neighbor Zero-Shot Inference**  
[`Paper`](https://arxiv.org/abs/2205.13792) [`Code`](https://github.com/swj0419/kNN_prompt) `EMNLP` `Application` ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F563a851106623b9f112d0e2a290d3950a871079c%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)

**2021**
- **Efficient Nearest Neighbor Language Models**  
[`Paper`](https://arxiv.org/abs/2109.04212) [`Code`](https://github.com/jxhe/efficient-knnlm) `EMNLP` `Architecture` ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F0c47eb31b2dd76d8dc986173a1d3f00da1c9c74d%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)

- **Mention Memory: incorporating textual knowledge into Transformers through entity mention attention**  
[`Paper`](https://arxiv.org/abs/2110.06176) [`Code`](https://github.com/google-research/language/tree/master/language/mentionmemory) `arXiv` `Architecture` ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F7b7416c90e8d3fc9ad5c9fb3923a638f69294ed7%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)

**2020**

- ⭐ **REALM: Retrieval-Augmented Language Model Pre-Training**  
[`Paper`](https://arxiv.org/abs/2002.08909) [`Code`](https://github.com/google-research/language/blob/master/language/realm/README.md) [`HuggingFace`](https://huggingface.co/docs/transformers/model_doc/realm) `PMLR` `Architecture` ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F832fff14d2ed50eb7969c4c4b976c35776548f56%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)

- **Generalization through Memorization: Nearest Neighbor Language Models**  
[`Paper`](https://arxiv.org/abs/1911.00172) [`Code`](https://github.com/urvashik/knnlm) `ICLR` `Architecture` ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F7be8c119dbe065c52125ee7716601751f3116844%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)

- **Entities as Experts: Sparse Memory Access with Entity Supervision**  
[`Paper`](https://arxiv.org/abs/2004.07202) `EMNLP`  `Architecture` ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F016368185723d0ec99aafa4b5927300590d0647f%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)

- **Dense Passage Retrieval for Open-Domain Question Answering**  
[`Paper`](https://arxiv.org/abs/2004.04906) [`Code`](https://github.com/facebookresearch/DPR) `EMNLP` `Training` ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fb26f2037f769d5ffc5f7bdcec2de8da28ec14bee%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)

- ⭐ **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**  
[`Paper`](https://arxiv.org/abs/2005.11401) [`HuggingFace`](https://huggingface.co/facebook/rag-token-nq) `NeurIPS` ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F58ed1fbaabe027345f7bb3a6312d41c5aac63e22%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)
## Resources

### Lectures, Talks, Tutorials

**2024**

- **Modular RAG and RAG Flow**  
    *Yunfan Gao* (2024) `Tutorial`  
    [`Blog I`](https://medium.com/@yufan1602/modular-rag-and-rag-flow-part-%E2%85%B0-e69b32dc13a3)
    [`Blog II`](https://medium.com/@yufan1602/modular-rag-and-rag-flow-part-ii-77b62bf8a5d3)


**2023**

- **Stanford CS25: V3 I Retrieval Augmented Language Models**  
  *Douwe Kiela* (2023) `Lecture`  
  [`Video`](https://www.youtube.com/watch?v=mE7IDf2SmJg&ab_channel=StanfordOnline)

- **Building RAG-based LLM Applications for Production**  
  *Anyscale* (2023) `Tutorial`  
  [`Blog`](https://www.anyscale.com/blog/a-comprehensive-guide-for-building-rag-based-llm-applications-part-1)

- **Multi-Vector Retriever for RAG on tables, text, and images**  
  *LangChain* (2023) `Tutorial`  
  [`Blog`](https://blog.langchain.dev/semi-structured-multi-modal-rag)

- **Retrieval-based Language Models and Applications**  
  *Asai et al.* (2023) `Tutorial`  `ACL`  
  [`Website`](https://acl2023-retrieval-lm.github.io/) [`Video`](https://us06web.zoom.us/rec/play/6fqU9YDLoFtWqpk8w8I7oFrszHKW6JkbPVGgHsdPBxa69ecgCxbmfP33asLU3DJ74q5BXqDGR2ycOTFk.93teqylfi_uiViNK?canPlayFromShare=true&from=share_recording_detail&continueMode=true&componentName=rec-play&originRequestUrl=https%3A%2F%2Fus06web.zoom.us%2Frec%2Fshare%2FNrYheXPtE5zOlbogmdBg653RIu7RBO1uAsYH2CZt_hacD1jOHksRahGlERHc_Ybs.KGX1cRVtJBQtJf0o)

- **Advanced RAG Techniques: an Illustrated Overview**  
  *Ivan Ilin* (2023) `Tutorial`   
  [`Blog`](https://towardsai.net/p/machine-learning/advanced-rag-techniques-an-illustrated-overview)

- **Retrieval Augmented Language Modeling**  
  *Melissa Dell* (2023) `Lecture`  
  [`Video`](https://www.youtube.com/watch?v=XC4eFiIMOmY)


### Workshops

**2025**

- **MAGMaR - The 1st Workshop on Multimodal Augmented Generation via MultimodAl Retrieval**  
  *Reno Kriz, Kenton Murray, Eugene Yang, Francis Ferraro, Kate Sanders, Cameron Carpenter, Benjamin Van Durme*   
  @`ACL` 2025, July 31st - August 1st  
  [`Website`](https://nlp.jhu.edu/magmar)


**2024**

- **Towards Knowledgeable Language Models**  
  *Zoey Sha Li, Manling Li, Michael JQ Zhang, Eunsol Choi, Mor Geva, Peter Hase*   
  @`ACL` 2024, August 12-17  
  [`Website`](https://knowledgeable-lm.github.io/)


## Tools

<!-- Tool Template
- **Name**  
  *Description*  
  [`Website`](Link)
-->

- **LangChain**  
  *LangChain is a framework for developing applications powered by language models.*  
  [`Website`](https://www.langchain.com/)

- **LlamaIndex**  
  *LlamaIndex is a simple, flexible data framework for connecting custom data sources to large language models.*  
  [`Website`](https://www.llamaindex.ai/)

- **Verba**  
  *Verba is an open-source application designed to offer an end-to-end, streamlined, and user-friendly interface for Retrieval-Augmented Generation (RAG) out of the box.*  
  [`Website`](https://verba.weaviate.io/)

- **NEUM**  
  *Open-source RAG framework optimized for large-scale and real-time data.*  
  [`Website`](https://www.neum.ai/)

- **Kiln**  
  *Open-source application to build and evaluate production-ready RAG pipelines.*  
  [`Website`](https://kiln.tech) [`GitHub`](https://github.com/Kiln-AI/Kiln)

- **Unstructured**  
  *Unstructured.io offers a powerful toolkit that handles the ingestion and data preprocessing step, allowing you to focus on the more exciting downstream steps in your machine learning pipeline. Unstructured has over a dozen data connectors that easily integrate with various data sources, including AWS S3, Discord, Slack, Wikipedia, and more.*  
  [`Website`](https://unstructured.io/)

- **CocoIndex**  
  *CocoIndex is an open-source ETL framework to index data for AI, such as RAG; with realtime incremental updates and support custom logic like lego.*  
  [`Website`](https://cocoindex.io/)

- **Agentset**  
  *Open-source production-ready RAG platform with built-in agentic reasoning, hybrid search, and multimodal support.*  
  [`Website`](https://agentset.ai/) [`GitHub`](https://github.com/agentset-ai/agentset)

## Other Collections

- [Awesome LLM RAG](https://github.com/jxzhangjhu/Awesome-LLM-RAG)
- [Awesome RAG](https://github.com/frutik/Awesome-RAG)
- [Awesome LLM with RAG](https://github.com/HKUST-AI-Lab/Awesome-LLM-with-RAG)
- [RAG-Survey](https://github.com/Tongji-KGLLM/RAG-Survey)
- [Awesome LLM Reader](https://github.com/HITsz-TMG/awesome-llm-reader): A Repository of Retrieval-augmented LLMs

