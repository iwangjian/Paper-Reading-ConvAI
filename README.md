# Paper-Reading
Paper reading list in natural language processing (NLP), with special emphasis on dialogue systems, text generation, and related topics. This repo will keep updating 🤗 ...

- [Deep Learning in NLP](#deep-learning-in-nlp)
- [Pre-trained Language Models](#pre-trained-language-models)
- [Natural Language Generation](#natural-language-generation)
- [Dialogue System](#dialogue-system)
  - [Rec-oriented Dialogue (CRS)](#rec-oriented-dialogue-crs)
  - [Knowledge-Grounded Dialogue](#knowledge-grounded-dialogue)
  - [Task-Oriented Dialogue](#task-oriented-dialogue)
  - [Open-domain Dialogue](#open-domain-dialogue)
  - [Personalized Dialogue](#personalized-dialogue)
  - [Dialogue Evaluation](#dialogue-evaluation)
  - [Misc](#misc)
- [Text Summarization](#text-summarization)
- [Knowledge Representation Learning](#knowledge-representation-learning)
- [Machine Translation](#machine-translation)
- [Question Answering](#question-answering)
- [Reading Comprehension](#reading-comprehension)
- [Image Captioning](#image-captioning)
- [Text Matching](#text-matching)

***

## Deep Learning in NLP
* **Data Augmentation**: "A Survey of Data Augmentation Approaches for NLP". ACL-Findings(2021) [[PDF]](http://arxiv.org/abs/2105.03075)
* **Survey of Transformers**: "A Survey of Transformers". arXiv(2021) [[PDF]](https://arxiv.org/abs/2106.04554) :star::star::star:
* **VAE**: "An Introduction to Variational Autoencoders". arXiv(2019) [[PDF]](https://arxiv.org/pdf/1906.02691.pdf)
* **Transformer**: "Attention is All you Need". NeurIPS(2017) [[PDF]](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) [[code-official]](https://github.com/tensorflow/tensor2tensor) [[code-tf]](https://github.com/Kyubyong/transformer) [[code-py]](https://github.com/jadore801120/attention-is-all-you-need-pytorch)
* **Transformer-XL**: "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context". ACL(2019) [[PDF]](https://www.aclweb.org/anthology/P19-1285) [[code]](https://github.com/kimiyoung/transformer-xl)
* **ConvS2S**: "Convolutional Sequence to Sequence Learning". ICML(2017) [[PDF]](http://proceedings.mlr.press/v70/gehring17a/gehring17a.pdf)
* **Survey of Attention**: "An Introductory Survey on Attention Mechanisms in NLP Problems". arXiv(2018) [[PDF]](https://arxiv.org/pdf/1811.05544.pdf) :star::star::star::star::star:
* **Additive Attention**: "Neural Machine Translation by Jointly Learning to Align and Translate". ICLR(2015) [[PDF]](https://arxiv.org/pdf/1409.0473.pdf) 
* **Multiplicative Attention**: "Effective Approaches to Attention-based Neural Machine Translation". EMNLP(2015) [[PDF]](https://www.aclweb.org/anthology/D15-1166)
* **Memory Net**: "End-To-End Memory Networks". NeurIPS(2015) [[PDF]](http://papers.nips.cc/paper/5846-end-to-end-memory-networks.pdf)
* **Pointer Net**: "Pointer Networks". NeurIPS(2015) [[PDF]](http://papers.nips.cc/paper/5866-pointer-networks.pdf) 
* **Copying Mechanism**: "Incorporating Copying Mechanism in Sequence-to-Sequence Learning". ACL(2016) [[PDF]](https://www.aclweb.org/anthology/P16-1154)
* **Coverage Mechanism**: "Modeling Coverage for Neural Machine Translation". ACL(2016) [[PDF]](https://www.aclweb.org/anthology/P16-1008)
* **GAN**: "Generative Adversarial Nets". NeurIPS(2014) [[PDF]](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)
* **SeqGAN**: "SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient". AAAI(2017) [[PDF]](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14344/14489) [[code]](https://github.com/LantaoYu/SeqGAN)
* **word2vec**: "word2vec Parameter Learning Explained". arXiv(2016) [[PDF]](https://arxiv.org/pdf/1411.2738.pdf) :star::star::star::star::star:
* **Glove**: "GloVe: Global Vectors for Word Representation". EMNLP(2014) [[PDF]](https://www.aclweb.org/anthology/D14-1162.pdf) [[code]](https://github.com/stanfordnlp/GloVe)
* **ELMo**: "Deep contextualized word representations". NAACL(2018) [[PDF]](https://www.aclweb.org/anthology/N18-1202) [[code]](https://github.com/allenai/bilm-tf)
* **Multi-task Learning**: "An Overview of Multi-Task Learning in Deep Neural Networks". arXiv(2017) [[PDF]](https://arxiv.org/pdf/1706.05098.pdf)
* **Gradient Descent**: "An Overview of Gradient Descent Optimization Algorithms". arXiv(2016) [[PDF]](https://arxiv.org/pdf/1609.04747.pdf) :star::star::star::star::star:


## Pre-trained Language Models
* **Survey of PLMs**: "Pre-Trained Models: Past, Present and Future". arXiv(2021) [[code]](http://arxiv.org/abs/2106.07139)
* **GLM**: "All NLP Tasks Are Generation Tasks: A General Pretraining Framework". arXiv(2021) [[PDF]](http://arxiv.org/abs/2103.10360) [[code]](https://github.com/THUDM/GLM)
* **BERTSeq2Seq**: "Leveraging Pre-trained Checkpoints for Sequence Generation Tasks". TACL(2020) [[PDF]](https://arxiv.org/pdf/1907.12461.pdf) [[code-tf]](https://github.com/google-research/google-research/tree/master/bertseq2seq) [[code-py]](https://github.com/huggingface/transformers)
* **PALM**: "PALM: Pre-training an Autoencoding&Autoregressive Language Model for Context-conditioned Generation". EMNLP(2020) [[PDF]](https://www.aclweb.org/anthology/2020.emnlp-main.700.pdf) [[code]](https://github.com/alibaba/AliceMind)
* **BART**: "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension". ACL(2020) [[PDF]](https://www.aclweb.org/anthology/2020.acl-main.703.pdf) [[code]](https://github.com/huggingface/transformers)
* **ERNIE-GEN**: "ERNIE-GEN: An Enhanced Multi-Flow Pre-training and Fine-tuning Framework for Natural Language Generation". IJCAI(2020) [[PDF]](https://arxiv.org/pdf/2001.11314.pdf) [[code]](https://github.com/PaddlePaddle/ERNIE/tree/repro/ernie-gen) :star::star::star:
* **T5**: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer". JMLR(2020) [[PDF]](http://arxiv.org/abs/1910.10683) [[code-tf]](https://github.com/google-research/text-to-text-transfer-transformer) [[code-py]](https://github.com/huggingface/transformers)
* **MASS**: "MASS: Masked Sequence to Sequence Pre-training for Language Generation". ICML(2019) [[PDF]](http://arxiv.org/abs/1905.02450) [[code]](https://github.com/microsoft/MASS)
* **PLMs**: "Pre-trained Models for Natural Language Processing: A Survey". arXiv(2020) [[PDF]](https://arxiv.org/pdf/2003.08271.pdf)
* **ALBERT**: "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations". ICLR(2020) [[PDF]](https://openreview.net/pdf?id=H1eA7AEtvS)
* **TinyBERT**: "TinyBERT: Distilling BERT for Natural Language Understanding". arXiv(2019) [[PDF]](https://arxiv.org/pdf/1909.10351.pdf) [[code]](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT)
* **Chinese BERT**: "Pre-Training with Whole Word Masking for Chinese BERT". arXiv(2019) [[PDF]](https://arxiv.org/pdf/1906.08101.pdf) [[code]](https://github.com/ymcui/Chinese-BERT-wwm)
* **SpanBERT**: "SpanBERT: Improving Pre-training by Representing and Predicting Spans". TACL(2020) [[PDF]](https://arxiv.org/pdf/1907.10529.pdf) [[code]](https://github.com/facebookresearch/SpanBERT)
* **RoBERTa**: "RoBERTa: A Robustly Optimized BERT Pretraining Approach". arXiv(2019) [[PDF]](https://arxiv.org/pdf/1907.11692.pdf) [[code]](https://github.com/pytorch/fairseq)
* **UniLM**: "Unified Language Model Pre-training for
Natural Language Understanding and Generation". NeurIPS(2019) [[PDF]](http://papers.nips.cc/paper/9464-unified-language-model-pre-training-for-natural-language-understanding-and-generation.pdf) [[code]](https://github.com/microsoft/unilm) :star::star::star::star::star:
* **XLNet**: "XLNet: Generalized Autoregressive Pretraining for Language Understanding". NeurIPS(2019) [[PDF]](http://papers.nips.cc/paper/8812-xlnet-generalized-autoregressive-pretraining-for-language-understanding.pdf) [[code]](https://github.com/zihangdai/xlnet)
* **XLM**: "Cross-lingual Language Model Pretraining". NeurIPS(2019) [[PDF]](http://papers.nips.cc/paper/8928-cross-lingual-language-model-pretraining.pdf) [[code]](https://github.com/facebookresearch/XLM)
* **BERT**: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding". NAACL(2019) [[PDF]](https://www.aclweb.org/anthology/N19-1423) [[code]](https://github.com/google-research/bert) :star::star::star::star::star:


## Natural Language Generation
* **FastSeq**: "EL-Attention: Memory Efficient Lossless Attention for Generation". ICML(2021) [[PDF]](http://arxiv.org/abs/2105.04779) [[code]](https://github.com/microsoft/fastseq) :star::star::star:
* **Repetition-Problem-NLG**: "A Theoretical Analysis of the Repetition Problem in Text Generation". AAAI(2021) [[PDF]](https://arxiv.org/pdf/2012.14660.pdf) [[code]](https://github.com/fuzihaofzh/repetition-problem-nlg)
* **CoMMA**: "A Study of Non-autoregressive Model for Sequence Generation". ACL(2020) [[PDF]](https://www.aclweb.org/anthology/2020.acl-main.15.pdf)
* **Nucleus Sampling**: "The Curious Case of Neural Text Degeneration". ICLR(2020) [[PDF]](https://openreview.net/forum?id=rygGQyrFvH) [[code]](https://github.com/ari-holtzman/degen) :star::star::star:
* **Cascaded Generation**: "Cascaded Text Generation with Markov Transformers". NeurIPS(2020) [[PDF]](https://arxiv.org/pdf/2006.01112.pdf) [[code]](https://github.com/harvardnlp/cascaded-generation)
* **Sequence Generation**: "A Generalized Framework of Sequence Generation with Application to Undirected Sequence Models". arXiv(2019) [[PDF]](https://arxiv.org/pdf/1905.12790.pdf) [[code]](https://github.com/nyu-dl/dl4mt-seqgen)
* **Entmax**: "Sparse Sequence-to-Sequence Models". ACL(2019) [[PDF]](https://www.aclweb.org/anthology/P19-1146) [[code]](https://github.com/deep-spin/entmax)


## Dialogue System

### Rec-oriented Dialogue (CRS)
* **CRS-Survey**: "Advances and Challenges in Conversational Recommender Systems: A Survey
". arXiv(2021) [[PDF]](https://arxiv.org/pdf/2101.09459.pdf)
* **CRS-Lab**: "CRSLab: An Open-Source Toolkit for Building Conversational Recommender System". arXiv(2021) [[PDF]](https://arxiv.org/pdf/2101.00939.pdf) [[code]](https://github.com/RUCAIBox/CRSLab) :star::star::star:
* **CR-Walker**: "Bridging the Gap between Conversational Reasoning and Interactive Recommendation". arXiv(2020) [[PDF]](https://arxiv.org/pdf/2010.10333.pdf) [[code]](https://github.com/truthless11/CR-Walker)
* **INSPIRED**: "INSPIRED: Toward Sociable Recommendation Dialog Systems". EMNLP(2020) [[PDF]](https://www.aclweb.org/anthology/2020.emnlp-main.654.pdf) [[data]](https://github.com/sweetpeach/Inspired)
* **TG-ReDial**: "Towards Topic-Guided Conversational Recommender System". COLING(2020) [[PDF]](https://www.aclweb.org/anthology/2020.coling-main.365.pdf) [[code]](https://github.com/RUCAIBox/TG-ReDial)
* **DuRecDial**: "Towards Conversational Recommendation over Multi-Type Dialogs". ACL(2020) [[PDF]](https://arxiv.org/pdf/2005.03954.pdf) [[code]](https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2020-DuRecDial) :star::star::star:
* **KGSF**: "Improving Conversational Recommender Systems via Knowledge Graph based Semantic Fusion". KDD(2020) [[PDF]](https://arxiv.org/pdf/2007.04032.pdf) [[code]](https://github.com/RUCAIBox/KGSF)
* **KBRD**: "Towards Knowledge-Based Recommender Dialog System". EMNLP(2019) [[PDF]](https://www.aclweb.org/anthology/D19-1189.pdf) [[code]](https://github.com/THUDM/KBRD)
* **GoRecDial**: "Recommendation as a Communication Game: Self-Supervised Bot-Play for Goal-oriented Dialogue". EMNLP(2019) [[PDF]](https://www.aclweb.org/anthology/D19-1203.pdf) [[code]](https://github.com/facebookresearch/ParlAI)
* **ReDial**: "Towards Deep Conversational Recommendations". NeurIPS(2018) [[PDF]](http://papers.nips.cc/paper/8180-towards-deep-conversational-recommendations.pdf) [[data]](https://github.com/ReDialData/website)


### Knowledge-Grounded Dialogue
* **KnowledGPT**: "Knowledge-Grounded Dialogue Generation with Pre-trained Language Models". EMNLP(2020) [[PDF]](https://www.aclweb.org/anthology/2020.emnlp-main.272)
* **DiffKS**: "Difference-aware Knowledge Selection for Knowledge-grounded Conversation Generation". EMNLP-Findings(2020) [[PDF]](https://www.aclweb.org/anthology/2020.findings-emnlp.11) [[code]](https://github.com/chujiezheng/DiffKS)
* **DukeNet**: "DukeNet: A Dual Knowledge Interaction Network for Knowledge-Grounded Conversation". SIGIR(2020) [[PDF]](https://dl.acm.org/doi/10.1145/3397271.3401097) [[code]](https://github.com/ChuanMeng/DukeNet)
* **CCN**: "Cross Copy Network for Dialogue Generation". EMNLP(2020) [[PDF]](https://www.aclweb.org/anthology/2020.emnlp-main.149) [[code]](https://github.com/jichangzhen/CCN)
* **PIPM**: "Bridging the Gap between Prior and Posterior Knowledge Selection for Knowledge-Grounded Dialogue Generation". EMNLP(2020) [[PDF]](https://www.aclweb.org/anthology/2020.emnlp-main.275)
* **ConKADI**: "Diverse and Informative Dialogue Generation with Context-Specific Commonsense Knowledge Awareness". ACL(2020) [[PDF]](https://www.aclweb.org/anthology/2020.acl-main.515) [[code]](https://github.com/pku-orangecat/ACL2020-ConKADI) :star::star::star::star::star:
* **KIC**: "Generating Informative Conversational Response using Recurrent Knowledge-Interaction and Knowledge-Copy". ACL(2020) [[PDF]](https://www.aclweb.org/anthology/2020.acl-main.6)
* **SKT**: "Sequential Latent Knowledge Selection for Knowledge-Grounded Dialogue". ICLR(2020) [[PDF]](https://openreview.net/pdf?id=Hke0K1HKwr) [[code]](https://github.com/bckim92/sequential-knowledge-transformer) :star::star::star::star:
* **KdConv**: "KdConv: A Chinese Multi-domain Dialogue Dataset Towards Multi-turn Knowledge-driven Conversation". ACL(2020) [[PDF]](https://arxiv.org/pdf/2004.04100.pdf) [[data]](https://github.com/thu-coai/KdConv)
* **RefNet**: "RefNet: A Reference-aware Network for Background Based Conversation". AAAI(2020) [[PDF]](https://arxiv.org/pdf/1908.06449.pdf) [[code]](https://github.com/ChuanMeng/RefNet)
* **GLKS**: "Thinking Globally, Acting Locally: Distantly Supervised Global-to-Local Knowledge Selection for Background Based Conversation". AAAI(2020) [[PDF]](https://arxiv.org/pdf/1908.09528.pdf) [[code]](https://github.com/PengjieRen/GLKS)
* **DuConv**: "Proactive Human-Machine Conversation with Explicit Conversation Goals". ACL(2019) [[PDF]](https://www.aclweb.org/anthology/P19-1369) [[code]](https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2019-DuConv)
* **OpenDialKG**: "OpenDialKG: Explainable Conversational Reasoning with Attention-based Walks over Knowledge Graphs". ACL(2019) [[PDF]](https://www.aclweb.org/anthology/P19-1081) [[data]](https://github.com/facebookresearch/opendialkg)
* **PostKS**: "Learning to Select Knowledge for Response Generation in Dialog Systems". IJCAI(2019) [[PDF]](https://www.ijcai.org/proceedings/2019/0706.pdf)
* **Two-Stage-Transformer**: "Wizard of Wikipedia: Knowledge-Powered Conversational agents". ICLR(2019) [[PDF]](https://arxiv.org/pdf/1811.01241.pdf) 
* **NKD**: "Knowledge Diffusion for Neural Dialogue Generation". ACL(2018) [[PDF]](https://www.aclweb.org/anthology/P18-1138) [[data]](https://github.com/liushuman/neural-knowledge-diffusion) 
* **Dual Fusion**: "Smarter Response with Proactive Suggestion: A New Generative Neural Conversation Paradigm". IJCAI(2018) [[PDF]](https://www.ijcai.org/proceedings/2018/0629.pdf)
* **CCM**: "Commonsense Knowledge Aware Conversation Generation with Graph Attention". IJCAI(2018) [[PDF]](https://www.ijcai.org/proceedings/2018/0643.pdf) [[code]](https://github.com/tuxchow/ccm) :star::star::star::star::star:
* **MTask**: "A Knowledge-Grounded Neural Conversation Model". AAAI(2018)  [[PDF]](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16710/16057)
* **GenDS**: "Flexible End-to-End Dialogue System for Knowledge Grounded Conversation". arXiv(2017) [[PDF]](https://arxiv.org/pdf/1709.04264.pdf)


### Task-Oriented Dialogue
* **ToD-BERT**: "ToD-BERT: Pre-trained Natural Language Understanding for Task-Oriented Dialogues". arXiv(2020) [[PDF]](https://arxiv.org/pdf/2004.06871.pdf) [[code]](https://github.com/jasonwu0731/ToD-BERT) 
* **DDMN**: "Dual Dynamic Memory Network for End-to-End Multi-turn Task-oriented Dialog Systems". COLING(2020) [[PDF]](https://www.aclweb.org/anthology/2020.coling-main.362/) [[code]](https://github.com/siat-nlp/DDMN) :star::star::star:
* **GraphDialog**: "GraphDialog: Integrating Graph Knowledge into End-to-End Task-Oriented Dialogue Systems". EMNLP(2020) [[PDF]](https://www.aclweb.org/anthology/2020.emnlp-main.147) [[code]](https://github.com/shiquanyang/GraphDialog)
* **DF-Net**: "Dynamic Fusion Network for Multi-Domain End-to-end Task-Oriented Dialog". ACL(2020) [[PDF]](https://arxiv.org/pdf/2004.11019.pdf) [[code]](https://github.com/LooperXX/DF-Net)
* **MALA**: "MALA: Cross-Domain Dialogue Generation with Action Learning". AAAI(2020) [[PDF]](https://arxiv.org/pdf/1912.08442.pdf)
* **CrossWOZ**: "CrossWOZ: A Large-Scale Chinese Cross-Domain Task-Oriented Dialogue Dataset". TACL(2020) [[PDF]](https://arxiv.org/pdf/2002.11893.pdf) [[code]](https://github.com/thu-coai/CrossWOZ) 
* **MultiWOZ**: "MultiWOZ - A Large-Scale Multi-Domain Wizard-of-Oz Dataset for Task-Oriented Dialogue Modelling". EMNLP(2018) [[PDF]](https://www.aclweb.org/anthology/D18-1547) [[code]](https://github.com/budzianowski/multiwoz)
* **Neural Task-Oriented Dialogue**: "Learning to Memorize in Neural Task-Oriented Dialogue Systems". MPhil Thesis(2019) [[PDF]](https://arxiv.org/pdf/1905.07687.pdf) :star::star::star::star:
* **GLMP**: "Global-to-local Memory Pointer Networks for Task-Oriented Dialogue". ICLR(2019) [[PDF]](https://arxiv.org/pdf/1901.04713.pdf) [[code]](https://github.com/jasonwu0731/GLMP) :star::star::star::star::star:
* **KB Retriever**: "Entity-Consistent End-to-end Task-Oriented Dialogue System with KB Retriever". EMNLP(2019) [[PDF]](https://www.aclweb.org/anthology/D19-1013.pdf) [[data]](https://github.com/yizhen20133868/Retriever-Dialogue)
* **TRADE**: "Transferable Multi-Domain State Generator for Task-Oriented
Dialogue Systems". ACL(2019) [[PDF]](https://www.aclweb.org/anthology/P19-1078) [[code]](https://github.com/jasonwu0731/trade-dst)
* **WMM2Seq**: "A Working Memory Model for Task-oriented Dialog Response Generation". ACL(2019) [[PDF]](https://www.aclweb.org/anthology/P19-1258)
* **Pretrain-Fine-tune**: "Training Neural Response Selection for Task-Oriented Dialogue Systems". ACL(2019) [[PDF]](https://www.aclweb.org/anthology/P19-1536) [[data]](https://github.com/PolyAI-LDN/conversational-datasets)
* **Multi-level Mem**: "Multi-Level Memory for Task Oriented Dialogs". NAACL(2019) [[PDF]](https://www.aclweb.org/anthology/N19-1375) [[code]](https://github.com/DineshRaghu/multi-level-memory-network)  :star::star::star:
* **BossNet**: "Disentangling Language and Knowledge in Task-Oriented Dialogs
". NAACL(2019) [[PDF]](https://www.aclweb.org/anthology/N19-1126) [[code]](https://github.com/dair-iitd/BossNet)
* **SL+RL**: "Dialogue Learning with Human Teaching and Feedback in End-to-End Trainable Task-Oriented Dialogue Systems". NAACL(2018) [[PDF]](https://www.aclweb.org/anthology/N18-1187)
* **MAD**: "Memory-augmented Dialogue Management for Task-oriented Dialogue Systems". TOIS(2018) [[PDF]](https://arxiv.org/pdf/1805.00150.pdf) :star::star::star:
* **TSCP**: "Sequicity: Simplifying Task-oriented Dialogue Systems with Single Sequence-to-Sequence Architectures". ACL(2018) [[PDF]](https://www.aclweb.org/anthology/P18-1133) [[code]](https://github.com/WING-NUS/sequicity)
* **Mem2Seq**: "Mem2Seq: Effectively Incorporating Knowledge Bases into End-to-End Task-Oriented Dialog Systems". ACL(2018) [[PDF]](https://www.aclweb.org/anthology/P18-1136) [[code]](https://github.com/HLTCHKUST/Mem2Seq) :star::star::star::star:
* **Topic-Seg-Label**: "A Weakly Supervised Method for Topic Segmentation and Labeling in Goal-oriented Dialogues via Reinforcement Learning". IJCAI(2018) [[PDF]](https://www.ijcai.org/proceedings/2018/0612.pdf) [[code]](https://github.com/truthless11/Topic-Seg-Label)
* **AliMe**: "AliMe Chat: A Sequence to Sequence and Rerank based Chatbot Engine". ACL(2017) [[PDF]](https://aclweb.org/anthology/P17-2079)
* **KVR Net**: "Key-Value Retrieval Networks for Task-Oriented Dialogue". SIGDIAL(2017) [[PDF]](https://www.aclweb.org/anthology/W17-5506) [[data]](https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/)


### Open-domain Dialogue
* **DialogBERT**: "DialogBERT: Discourse-Aware Response Generation via Learning to Recover and Rank Utterances". AAAI(2021) [[PDF]](https://arxiv.org/pdf/2012.01775.pdf)
* **CDial-GPT**: "A Large-Scale Chinese Short-Text Conversation Dataset". NLPCC(2020) [[PDF]](https://arxiv.org/pdf/2008.03946.pdf) [[code]](https://github.com/thu-coai/CDial-GPT)
* **DialoGPT**: "DialoGPT : Large-Scale Generative Pre-training for Conversational Response Generation". ACL(2020) [[PDF]](https://arxiv.org/pdf/1911.00536.pdf) [[code]](https://github.com/microsoft/DialoGPT) 
* **PLATO**: "PLATO: Pre-trained Dialogue Generation Model with Discrete Latent Variable". ACL(2020) [[PDF]](https://arxiv.org/pdf/1910.07931.pdf) [[code]](https://github.com/PaddlePaddle/Research/tree/master/NLP/Dialogue-PLATO) 
* **Guyu**: "An Empirical Investigation of Pre-Trained Transformer Language Models for Open-Domain Dialogue Generation". arXiv(2020) [[PDF]](https://arxiv.org/pdf/2003.04195.pdf) [[code]](https://github.com/lipiji/Guyu)
* **CL4Dialogue**: "Group-wise Contrastive Learning for Neural Dialogue Generation". EMNLP-Findings(2020) [[PDF]](https://www.aclweb.org/anthology/2020.findings-emnlp.70) [[code]](https://github.com/hengyicai/ContrastiveLearning4Dialogue) :star::star::star:
* **Neg-train**: "Negative Training for Neural Dialogue Response Generation". ACL(2020) [[PDF]](https://www.aclweb.org/anthology/2020.acl-main.185) [[code]](https://github.mit.edu/tianxing/negativetraining_acl2020)
* **HDSA**: "Semantically Conditioned Dialog Response Generation via Hierarchical Disentangled Self-Attention". ACL(2019) [[PDF]](https://www.aclweb.org/anthology/P19-1360) [[code]](https://github.com/wenhuchen/HDSA-Dialog) :star::star::star:
* **CAS**: "Skeleton-to-Response: Dialogue Generation Guided by Retrieval Memory". NAACL(2019) [[PDF]](https://www.aclweb.org/anthology/N19-1124) [[code]](https://github.com/jcyk/Skeleton-to-Response)
* **Edit-N-Rerank**: "Response Generation by Context-aware Prototype Editing". AAAI(2019) [[PDF]](https://arxiv.org/pdf/1806.07042.pdf) [[code]](https://github.com/MarkWuNLP/ResponseEdit) :star::star::star:
* **HVMN**: "Hierarchical Variational Memory Network for Dialogue Generation". WWW(2018) [[PDF]](https://dl.acm.org/citation.cfm?doid=3178876.3186077) [[code]](https://github.com/chenhongshen/HVMN)
* **XiaoIce**: "The Design and Implementation of XiaoIce, an Empathetic Social Chatbot". arXiv(2018) [[PDF]](https://arxiv.org/pdf/1812.08989.pdf)
* **D2A**: "Dialog-to-Action: Conversational Question Answering Over a Large-Scale Knowledge Base". NeurIPS(2018) [[PDF]](http://papers.nips.cc/paper/7558-dialog-to-action-conversational-question-answering-over-a-large-scale-knowledge-base.pdf) [[code]](https://github.com/guoday/Dialog-to-Action)
* **DAIM**: "Generating Informative and Diverse Conversational Responses via Adversarial Information Maximization". NeurIPS(2018) [[PDF]](http://papers.nips.cc/paper/7452-generating-informative-and-diverse-conversational-responses-via-adversarial-information-maximization.pdf)
* **REASON**: "Dialog Generation Using Multi-turn Reasoning Neural Networks". NAACL(2018) [[PDF]](https://www.aclweb.org/anthology/N18-1186) 
* **STD/HTD**: "Learning to Ask Questions in Open-domain Conversational Systems with Typed Decoders". ACL(2018) [[PDF]](https://www.aclweb.org/anthology/P18-1204) [[code]](https://github.com/victorywys/Learning2Ask_TypedDecoder) 
* **CSF**: "Generating Informative Responses with Controlled Sentence Function". ACL(2018) [[PDF]](https://www.aclweb.org/anthology/P18-1139) [[code]](https://github.com/kepei1106/SentenceFunction)
* **DAWnet**: "Chat More: Deepening and Widening the Chatting Topic via A Deep Model". SIGIR(2018) [[PDF]](https://dl.acm.org/citation.cfm?doid=3209978.3210061) [[code]](https://sigirdawnet.wixsite.com/dawnet)
* **ZSDG**: "Zero-Shot Dialog Generation with Cross-Domain Latent Actions". SIGDIAL(2018) [[PDF]](https://www.aclweb.org/anthology/W18-5001) [[code]](https://github.com/snakeztc/NeuralDialog-ZSDG) 
* **DUA**: "Modeling Multi-turn Conversation with Deep Utterance Aggregation". COLING(2018) [[PDF]](https://www.aclweb.org/anthology/C18-1317) [[code]](https://github.com/cooelf/DeepUtteranceAggregation)
* **Data-Aug**: "Sequence-to-Sequence Data Augmentation for Dialogue Language Understanding". COLING(2018) [[PDF]](https://www.aclweb.org/anthology/C18-1105) [[code]](https://github.com/AtmaHou/Seq2SeqDataAugmentationForLU)
* **DC-MMI**: "Generating More Interesting Responses in Neural Conversation Models with Distributional Constraints". EMNLP(2018) [[PDF]](https://www.aclweb.org/anthology/D18-1431) [[code]](https://github.com/abaheti95/DC-NeuralConversation)
* **cVAE-XGate/CGate**: "Better Conversations by Modeling, Filtering, and Optimizing for Coherence and Diversity". EMNLP(2018) [[PDF]](https://www.aclweb.org/anthology/D18-1432) [[code]](https://github.com/XinnuoXu/CVAE_Dial)
* **Retrieval+multi-seq2seq**: "An Ensemble of Retrieval-Based and Generation-Based Human-Computer Conversation Systems". IJCAI(2018) [[PDF]](https://www.ijcai.org/proceedings/2018/0609.pdf)
* **DAM**: "Multi-Turn Response Selection for Chatbots with Deep Attention Matching Network". ACL(2018) [[PDF]](https://www.aclweb.org/anthology/P18-1103) [[code]](https://github.com/baidu/Dialogue/tree/master/DAM) :star::star::star::star:
* **SMN**: "Sequential Matching Network: A New Architecture for Multi-turn Response Selection in Retrieval-Based Chatbots". ACL(2017) [[PDF]](https://aclweb.org/anthology/P17-1046)  [[code]](https://github.com/MarkWuNLP/MultiTurnResponseSelection) :star::star::star:
* **CVAE/KgCVAE**: "Learning Discourse-level Diversity for Neural Dialog Models using Conditional Variational Autoencoders". ACL(2017) [[PDF]](https://aclweb.org/anthology/P17-1061) [[code]](https://github.com/snakeztc/NeuralDialog-CVAE) :star::star::star:
* **TA-Seq2Seq**: "Topic Aware Neural Response Generation". AAAI(2017) [[PDF]](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14563/14260) [[code]](https://github.com/LynetteXing1991/TA-Seq2Seq)
* **MA**: "Mechanism-Aware Neural Machine for Dialogue Response Generation". AAAI(2017) [[PDF]](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14471/14267)
* **VHRED**: "A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues". AAAI(2017) [[PDF]](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14567/14219) [[code]](https://github.com/julianser/hed-dlg-truncated)
* **HRED**: "Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models". AAAI(2016) [[PDF]](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/11957/12160) [[code]](https://github.com/julianser/hed-dlg)
* **RL-Dialogue**: "Deep Reinforcement Learning for Dialogue Generation". EMNLP(2016) [[PDF]](https://www.aclweb.org/anthology/D16-1127)
* **MMI**: "A Diversity-Promoting Objective Function for Neural Conversation Models". NAACL-HLT(2016)  [[PDF]](https://www.aclweb.org/anthology/N16-1014) [[code]](https://github.com/jiweil/Neural-Dialogue-Generation)


### Personalized Dialogue
* **PABST**: "Unsupervised Enrichment of Persona-grounded Dialog with Background Stories". ACL(2021) [[PDF]](http://arxiv.org/abs/2106.08364) [[code]](https://github.com/majumderb/pabst)
* **PAML**: "Personalizing Dialogue Agents via Meta-Learning". ACL(2019) [[PDF]](https://www.aclweb.org/anthology/P19-1542) [[code]](https://github.com/HLTCHKUST/PAML)
* **PCCM**: "Assigning Personality/Profile to a Chatting Machine for Coherent Conversation Generation". IJCAI(2018) [[PDF]](https://www.ijcai.org/proceedings/2018/0595.pdf) [[code]](https://github.com/qianqiao/AssignPersonality)
* **ECM**: "Emotional Chatting Machine: Emotional Conversation Generation with Internal and External Memory". AAAI(2018) [[PDF]](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16455/15753) [[code]](https://github.com/tuxchow/ecm)

### Dialogue Evaluation
* **Spot-the-Bot**: "Spot The Bot: A Robust and Efficient Framework for the Evaluation of Conversational Dialogue Systems". EMNLP(2020) [[PDF]](https://www.aclweb.org/anthology/2020.emnlp-main.326) [[code]](https://github.com/jderiu/spot-the-bot-code)
* **BLEURT**: "BLEURT: Learning Robust Metrics for Text Generation". ACL(2020) [[PDF]](https://www.aclweb.org/anthology/2020.acl-main.704) [[code]](https://github.com/google-research/bleurt)
* **GRADE**: "GRADE: Automatic Graph-Enhanced Coherence Metric for Evaluating Open-Domain Dialogue Systems". ACL(2020) [[PDF]](https://www.aclweb.org/anthology/2020.emnlp-main.742) [[code]](https://github.com/li3cmz/GRADE)
* **uBLEU**: "uBLEU: Uncertainty-Aware Automatic Evaluation Method for Open-Domain Dialogue Systems". ACL(2020) [[PDF]](https://www.aclweb.org/anthology/2020.acl-srw.27) [[code]](https://github.com/YumaTsuta/upsilon_bleu)
* **USR**: "USR: An Unsupervised and Reference Free Evaluation Metric for Dialog Generation". ACL(2020) [[PDF]](https://www.aclweb.org/anthology/2020.acl-main.64) [[code]](https://github.com/Shikib/usr)
* **ADVMT**: "One `Ruler` for All Languages: Multi-Lingual Dialogue Evaluation with Adversarial Multi-Task Learning". IJCAI(2018) [[PDF]](https://www.ijcai.org/proceedings/2018/0616.pdf)

### Misc
* **Survey of Dialogue**: "Recent Advances in Deep Learning Based Dialogue Systems: A Systematic Survey". arXiv(2021) [[PDF]](http://arxiv.org/abs/2105.04387) :star::star::star:
* **Survey of Dialogue**: "A Survey on Dialogue Systems: Recent Advances and New Frontiers". SIGKDD Explorations(2017) [[PDF]](https://arxiv.org/pdf/1711.01731.pdf)
* **Survey of Corpora**: "A Survey of Available Corpora For Building Data-Driven Dialogue Systems: The Journal Version". Dialogue & Discourse(2018) [[PDF]](http://dad.uni-bielefeld.de/index.php/dad/article/view/3690/3616)


## Knowledge Representation Learning
* **CoLAKE**: "CoLAKE: Contextualized Language and Knowledge Embedding". COLING(2020) [[PDF]](http://arxiv.org/abs/2010.00309) [[code]](https://github.com/txsun1997/CoLAKE)
* **KEPLER**: "KEPLER: A Unified Model for Knowledge Embedding and Pre-trained Language Representation". TACL(2020) [[PDF]](https://arxiv.org/pdf/1911.06136.pdf) [[code]](https://github.com/THU-KEG/KEPLER)
* **LUKE**: "LUKE: Deep Contextualized Entity Representations with Entity-aware Self-attention". EMNLP(2020) [[PDF]](https://www.aclweb.org/anthology/2020.emnlp-main.523) [[code]](https://github.com/studio-ousia/luke) :star::star::star:
* **GRF**: "Language Generation with Multi-Hop Reasoning on Commonsense Knowledge Graph". EMNLP(2020) [[PDF]](http://arxiv.org/abs/2009.11692) [[code]](https://github.com/cdjhz/multigen)
* **LM-as-KG**: "Language Models are Open Knowledge Graphs". arXiv(2020) [[PDF]](http://arxiv.org/abs/2010.11967)
* **LM-as-KB**: "Language Models as Knowledge Bases?". EMNLP(2019) [[PDF]](http://arxiv.org/abs/1909.01066) [[code]](https://github.com/facebookresearch/LAMA)
* **ERNIE(Tsinghua)**: "ERNIE: Enhanced Language Representation with Informative Entities". ACL(2019) [[PDF]](https://www.aclweb.org/anthology/P19-1139.pdf) [[code]](https://github.com/thunlp/ERNIE) :star::star::star:
* **ERNIE(Baidu)**: "ERNIE: Enhanced Representation through Knowledge Integration". arXiv(2019) [[PDF]](https://arxiv.org/pdf/1904.09223.pdf) [[code]](https://github.com/PaddlePaddle/ERNIE)


## Text Summarization
* **BERTSum**: "Fine-tune BERT for Extractive Summarization". arXiv(2019) [[PDF]](https://arxiv.org/pdf/1903.10318.pdf) [[code]](https://github.com/nlpyang/BertSum)
* **QASumm**: "Guiding Extractive Summarization with Question-Answering Rewards". NAACL(2019) [[PDF]](https://www.aclweb.org/anthology/N19-1264) [[code]](https://github.com/ucfnlp/summ_qa_rewards)
* **Re^3Sum**: "Retrieve, Rerank and Rewrite: Soft Template Based Neural Summarization". ACL(2018) [[PDF]](https://www.aclweb.org/anthology/P18-1015) [[code]](http://www4.comp.polyu.edu.hk/~cszqcao/data/IRSum_Resource.zip)
* **NeuSum**: "Neural Document Summarization by Jointly Learning to Score and Select Sentences". ACL(2018) [[PDF]](https://www.aclweb.org/anthology/P18-1061) 
* **rnn-ext+abs+RL+rerank**: "Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting". ACL(2018) [[PDF]](https://www.aclweb.org/anthology/P18-1063) [[Notes]](https://www.aclweb.org/anthology/attachments/P18-1063.Notes.pdf) [[code]](https://github.com/ChenRocks/fast_abs_rl) :star::star::star::star::star:
* **Seq2Seq+CGU**: "Global Encoding for Abstractive Summarization". ACL(2018) [[PDF]](https://www.aclweb.org/anthology/P18-2027) [[code]](https://github.com/lancopku/Global-Encoding)
* **ML+RL**: "A Deep Reinforced Model for Abstractive Summarization". ICLR(2018) [[PDF]](https://arxiv.org/pdf/1705.04304.pdf)
* **T-ConvS2S**: "Don’t Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization". EMNLP(2018) [[PDF]](https://www.aclweb.org/anthology/D18-1206) [[code]](https://github.com/shashiongithub/XSum)
* **RL-Topic-ConvS2S**: "A reinforced topic-aware convolutional sequence-to-sequence model for abstractive text summarization". IJCAI (2018) [[PDF]](https://www.ijcai.org/proceedings/2018/0619.pdf)
* **GANsum**: "Generative Adversarial Network for Abstractive Text Summarization". AAAI(2018) [[PDF]](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16238/16492)
* **FTSum**: "Faithful to the Original: Fact Aware Neural Abstractive Summarization". AAAI(2018) [[PDF]](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16121/16007)
* **PGN**: "Get To The Point: Summarization with Pointer-Generator Networks". ACL(2017) [[PDF]](https://aclweb.org/anthology/P17-1099) [[code]](https://github.com/abisee/pointer-generator) :star::star::star::star::star:
* **ABS/ABS+**: "A Neural Attention Model for Abstractive Sentence Summarization". EMNLP(2015) [[PDF]](https://www.aclweb.org/anthology/D15-1044)
* **RAS-Elman/RAS-LSTM**: "Abstractive Sentence Summarization with Attentive Recurrent Neural Networks". NAACL(2016) [[PDF]](https://www.aclweb.org/anthology/N16-1012) [[code]](https://github.com/facebookarchive/NAMAS)


## Machine Translation
* **Multi-pass decoder**: "Adaptive Multi-pass Decoder for Neural Machine Translation". EMNLP(2018) [[PDF]](https://www.aclweb.org/anthology/D18-1048) 
* **Deliberation Networks**: "Deliberation Networks: Sequence Generation Beyond One-Pass Decoding". NeurIPS(2017) [[PDF]](http://papers.nips.cc/paper/6775-deliberation-networks-sequence-generation-beyond-one-pass-decoding.pdf) :star::star::star:
* **KVMem-Attention**: "Neural Machine Translation with Key-Value Memory-Augmented Attention". IJCAI(2018) [[PDF]](https://www.ijcai.org/proceedings/2018/0357.pdf) :star::star::star:
* **Interactive-Attention**: "Interactive Attention for Neural Machine Translation". COLING(2016) [[PDF]](https://www.aclweb.org/anthology/C16-1205)


## Question Answering
* **CFC**: "Coarse-grain Fine-grain Coattention Network for Multi-evidence Question Answering". ICLR(2019) [[PDF]](https://arxiv.org/pdf/1901.00603.pdf)
* **MTQA**: "Multi-Task Learning with Multi-View Attention for Answer Selection and Knowledge Base Question Answering". AAAI(2019) [[PDF]](https://aaai.org/ojs/index.php/AAAI/article/view/4593) [[code]](https://github.com/dengyang17/MTQA)
* **CQG-KBQA**: "Knowledge Base Question Answering via Encoding of Complex Query Graphs". EMNLP(2018) [[PDF]](https://www.aclweb.org/anthology/D18-1242) [[code]](http://202.120.38.146/CompQA/) :star::star::star::star::star:
* **HR-BiLSTM**: "Improved Neural Relation Detection for Knowledge Base Question Answering". ACL(2017) [[PDF]](https://aclweb.org/anthology/P17-1053)
* **KBQA-CGK**: "An End-to-End Model for Question Answering over Knowledge Base with Cross-Attention Combining Global Knowledge". ACL(2017) [[PDF]](https://aclweb.org/anthology/P17-1021)
* **KVMem**: "Key-Value Memory Networks for Directly Reading Documents". EMNLP(2016) [[PDF]](https://www.aclweb.org/anthology/D16-1147)


## Reading Comprehension
* **DecompRC**: "Multi-hop Reading Comprehension through Question Decomposition and Rescoring". ACL(2019) [[PDF]](https://www.aclweb.org/anthology/P19-1613) [[code]](https://github.com/shmsw25/DecompRC)
* **FlowQA**: "FlowQA: Grasping Flow in History for Conversational Machine Comprehension". ICLR(2019) [[PDF]](https://arxiv.org/pdf/1810.06683.pdf) [[code]](https://github.com/momohuang/FlowQA) :star::star::star:
*  **SDNet**: "SDNet: Contextualized Attention-based Deep Network for Conversational Question Answering". arXiv(2018) [[PDF]](https://arxiv.org/pdf/1812.03593.pdf) [[code]](https://github.com/microsoft/SDNet)
* **MacNet**: "MacNet: Transferring Knowledge from Machine Comprehension to Sequence-to-Sequence Models". NeurIPS(2018) [[PDF]](http://papers.nips.cc/paper/7848-macnet-transferring-knowledge-from-machine-comprehension-to-sequence-to-sequence-models.pdf)


## Image Captioning
* **MLAIC**: "A Multi-task Learning Approach for Image Captioning". IJCAI(2018) [[PDF]](https://www.ijcai.org/proceedings/2018/0168.pdf) [[code]](https://github.com/andyweizhao/Multitask_Image_Captioning)
* **Up-Down Attention**: "Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering". CVPR(2018) [[PDF]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Anderson_Bottom-Up_and_Top-Down_CVPR_2018_paper.pdf) :star::star::star:
* **SCST**: "Self-critical Sequence Training for Image Captioning". CVPR(2017) [[PDF]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8099614) :star::star::star:


## Text Matching
* **Poly-encoder**: "Poly-encoders: Architectures and Pre-training Strategies for Fast and Accurate Multi-sentence Scorings". ICLR(2020) [[PDF]](https://openreview.net/pdf?id=SkxgnnNFvH) [[code]](https://github.com/sfzhou5678/PolyEncoder)
* **AugSBERT**: "Augmented SBERT: Data Augmentation Method for Improving Bi-Encoders for Pairwise Sentence Scoring Tasks". arXiv(2020) [[PDF]](https://arxiv.org/pdf/2010.08240.pdf) [[code]](https://github.com/UKPLab/sentence-transformers)
* **SBERT**: "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks". EMNLP(2019) [[PDF]](https://arxiv.org/pdf/1908.10084.pdf) [[code]](https://github.com/UKPLab/sentence-transformers)