# Paper-Reading
Paper reading list in natural language processing (NLP), with special emphasis on **dialogue systems**, **natural language generation** and relevant topics. This repo will keep updating 🤗 ...

- [Deep Learning in NLP](#deep-learning-in-nlp)
- [Pre-trained Language Models](#pre-trained-language-models)
- [Knowledge Representation Learning](#knowledge-representation-learning)
- [Dialogue System](#dialogue-system)
  - [Survey](#survey)
  - [Recommendation Dialogue and CRS](#recommendation-dialogue-and-crs)
  - [Target-guided Dialogue](#target-guided-dialogue)
  - [Knowledge-grounded Dialogue](#knowledge-grounded-dialogue)
  - [Task-oriented Dialogue](#task-oriented-dialogue)
  - [Emotion-aware Dialogue](#emotion-aware-dialogue)
  - [Persona-grounded Dialogue](#persona-grounded-dialogue)
  - [Open-domain Dialogue](#open-domain-dialogue)
  - [Dialogue Evaluation](#dialogue-evaluation)
- [Natural Language Generation](#natural-language-generation)
  - [Survey for NLG](#survey-for-nlg)
  - [Text Planning](#text-planning)
  - [Language Generation](#language-generation)
- [Others](#others)
  - [Text Summarization](#text-summarization)
  - [Machine Translation](#machine-translation)
  - [Question Answering](#question-answering)
  - [Text Matching](#text-matching)

***

## Deep Learning in NLP
* **Data Augmentation**: "A Survey of Data Augmentation Approaches for NLP". ACL-Findings(2021) [[PDF]](https://arxiv.org/abs/2105.03075)
* **Survey of Transformers**: "A Survey of Transformers". arXiv(2021) [[PDF]](https://arxiv.org/abs/2106.04554) :star::star::star:
* **Graphormer**: "Do Transformers Really Perform Bad for Graph Representation?". NeurIPS(2021) [[PDF]](https://arxiv.org/abs/2106.05234) [[code]](https://github.com/Microsoft/Graphormer)
* **GAT**: "Graph Attention Networks". ICLR(2018) [[PDF]](https://arxiv.org/pdf/1710.10903.pdf) [[code-tf]](https://github.com/PetarV-/GAT) [[code-py]](https://github.com/Diego999/pyGAT)
* **Transformer-XL**: "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context". ACL(2019) [[PDF]](https://www.aclweb.org/anthology/P19-1285) [[code]](https://github.com/kimiyoung/transformer-xl)
* **Transformer**: "Attention is All you Need". NeurIPS(2017) [[PDF]](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) [[code-official]](https://github.com/tensorflow/tensor2tensor) [[code-tf]](https://github.com/Kyubyong/transformer) [[code-py]](https://github.com/jadore801120/attention-is-all-you-need-pytorch)
* **VAE**: "An Introduction to Variational Autoencoders". arXiv(2019) [[PDF]](https://arxiv.org/pdf/1906.02691.pdf)
* **ConvS2S**: "Convolutional Sequence to Sequence Learning". ICML(2017) [[PDF]](https://proceedings.mlr.press/v70/gehring17a/gehring17a.pdf)
* **Survey of Attention**: "An Introductory Survey on Attention Mechanisms in NLP Problems". arXiv(2018) [[PDF]](https://arxiv.org/pdf/1811.05544.pdf) :star::star::star::star::star:
* **Additive Attention**: "Neural Machine Translation by Jointly Learning to Align and Translate". ICLR(2015) [[PDF]](https://arxiv.org/pdf/1409.0473.pdf) 
* **Multiplicative Attention**: "Effective Approaches to Attention-based Neural Machine Translation". EMNLP(2015) [[PDF]](https://www.aclweb.org/anthology/D15-1166)
* **Memory Net**: "End-To-End Memory Networks". NeurIPS(2015) [[PDF]](https://papers.nips.cc/paper/5846-end-to-end-memory-networks.pdf)
* **Copy Mechanism (PGN)**: "Get To The Point: Summarization with Pointer-Generator Networks". ACL(2017) [[PDF]](https://aclweb.org/anthology/P17-1099) [[code]](https://github.com/abisee/pointer-generator) :star::star::star::star::star:
* **Copy Mechanism**: "Incorporating Copying Mechanism in Sequence-to-Sequence Learning". ACL(2016) [[PDF]](https://www.aclweb.org/anthology/P16-1154)
* **Coverage Mechanism**: "Modeling Coverage for Neural Machine Translation". ACL(2016) [[PDF]](https://www.aclweb.org/anthology/P16-1008)
* **ELMo**: "Deep contextualized word representations". NAACL(2018) [[PDF]](https://www.aclweb.org/anthology/N18-1202) [[code]](https://github.com/allenai/bilm-tf)
* **Glove**: "GloVe: Global Vectors for Word Representation". EMNLP(2014) [[PDF]](https://www.aclweb.org/anthology/D14-1162.pdf) [[code]](https://github.com/stanfordnlp/GloVe)
* **word2vec**: "word2vec Parameter Learning Explained". arXiv(2016) [[PDF]](https://arxiv.org/pdf/1411.2738.pdf) :star::star::star::star::star:
* **SeqGAN**: "SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient". AAAI(2017) [[PDF]](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14344/14489) [[code]](https://github.com/LantaoYu/SeqGAN)
* **GAN**: "Generative Adversarial Nets". NeurIPS(2014) [[PDF]](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)
* **Multi-task Learning**: "An Overview of Multi-Task Learning in Deep Neural Networks". arXiv(2017) [[PDF]](https://arxiv.org/pdf/1706.05098.pdf)
* **Gradient Descent**: "An Overview of Gradient Descent Optimization Algorithms". arXiv(2016) [[PDF]](https://arxiv.org/pdf/1609.04747.pdf) :star::star::star::star::star:


## Pre-trained Language Models
* **Survey of PLMs**: "Pre-Trained Models: Past, Present and Future". arXiv(2021) [[PDF]](https://arxiv.org/abs/2106.07139)
* **Survey of PLMs**: "Pre-trained Models for Natural Language Processing: A Survey". arXiv(2020) [[PDF]](https://arxiv.org/pdf/2003.08271.pdf)
* **CPT**: "CPT: A Pre-Trained Unbalanced Transformer for Both Chinese Language Understanding and Generation". arXiv(2021) [[PDF]](https://arxiv.org/abs/2109.05729) [[code]](https://github.com/fastnlp/CPT) :star::star::star:
* **GLM**: "All NLP Tasks Are Generation Tasks: A General Pretraining Framework". arXiv(2021) [[PDF]](https://arxiv.org/abs/2103.10360) [[code]](https://github.com/THUDM/GLM)
* **PALM**: "PALM: Pre-training an Autoencoding&Autoregressive Language Model for Context-conditioned Generation". EMNLP(2020) [[PDF]](https://www.aclweb.org/anthology/2020.emnlp-main.700.pdf) [[code]](https://github.com/alibaba/AliceMind)
* **BART**: "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension". ACL(2020) [[PDF]](https://www.aclweb.org/anthology/2020.acl-main.703.pdf) [[code]](https://github.com/huggingface/transformers) :star::star::star:
* **T5**: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer". JMLR(2020) [[PDF]](https://arxiv.org/abs/1910.10683) [[code-tf]](https://github.com/google-research/text-to-text-transfer-transformer) [[code-py]](https://github.com/huggingface/transformers)
* **MASS**: "MASS: Masked Sequence to Sequence Pre-training for Language Generation". ICML(2019) [[PDF]](https://arxiv.org/abs/1905.02450) [[code]](https://github.com/microsoft/MASS)
* **ALBERT**: "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations". ICLR(2020) [[PDF]](https://openreview.net/pdf?id=H1eA7AEtvS)
* **TinyBERT**: "TinyBERT: Distilling BERT for Natural Language Understanding". arXiv(2019) [[PDF]](https://arxiv.org/pdf/1909.10351.pdf) [[code]](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT)
* **Chinese BERT**: "Pre-Training with Whole Word Masking for Chinese BERT". arXiv(2019) [[PDF]](https://arxiv.org/pdf/1906.08101.pdf) [[code]](https://github.com/ymcui/Chinese-BERT-wwm)
* **SpanBERT**: "SpanBERT: Improving Pre-training by Representing and Predicting Spans". TACL(2020) [[PDF]](https://arxiv.org/pdf/1907.10529.pdf) [[code]](https://github.com/facebookresearch/SpanBERT)
* **RoBERTa**: "RoBERTa: A Robustly Optimized BERT Pretraining Approach". arXiv(2019) [[PDF]](https://arxiv.org/pdf/1907.11692.pdf) [[code]](https://github.com/pytorch/fairseq)
* **UniLM**: "Unified Language Model Pre-training for
Natural Language Understanding and Generation". NeurIPS(2019) [[PDF]](https://papers.nips.cc/paper/9464-unified-language-model-pre-training-for-natural-language-understanding-and-generation.pdf) [[code]](https://github.com/microsoft/unilm) :star::star::star::star:
* **XLNet**: "XLNet: Generalized Autoregressive Pretraining for Language Understanding". NeurIPS(2019) [[PDF]](https://papers.nips.cc/paper/8812-xlnet-generalized-autoregressive-pretraining-for-language-understanding.pdf) [[code]](https://github.com/zihangdai/xlnet)
* **XLM**: "Cross-lingual Language Model Pretraining". NeurIPS(2019) [[PDF]](https://papers.nips.cc/paper/8928-cross-lingual-language-model-pretraining.pdf) [[code]](https://github.com/facebookresearch/XLM)
* **BERT**: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding". NAACL(2019) [[PDF]](https://www.aclweb.org/anthology/N19-1423) [[code]](https://github.com/google-research/bert) :star::star::star::star::star:


## Knowledge Representation Learning
* **FKGE**: "Differentially Private Federated Knowledge Graphs Embedding". CIKM(2021) [[PDF]](https://arxiv.org/abs/2105.07615) [[code]](https://github.com/HKUST-KnowComp/FKGE) :star::star::star:
* **ERICA**: "ERICA: Improving Entity and Relation Understanding for Pre-trained Language Models via Contrastive Learning". ACL(2021) [[PDF]](https://arxiv.org/abs/2012.15022) [[code]](https://github.com/thunlp/ERICA)
* **JointGT**: "JointGT: Graph-Text Joint Representation Learning for Text Generation from Knowledge Graphs". ACL-Findings(2021) [[PDF]](https://aclanthology.org/2021.findings-acl.223) [[code]](https://github.com/thu-coai/JointGT)
* **K-Adapter**: "K-Adapter: Infusing Knowledge into Pre-Trained Models with Adapters". ACL-Findings(2021) [[PDF]](https://arxiv.org/abs/2002.01808) [[code]](https://github.com/microsoft/k-adapter)
* **CoLAKE**: "CoLAKE: Contextualized Language and Knowledge Embedding". COLING(2020) [[PDF]](https://arxiv.org/abs/2010.00309) [[code]](https://github.com/txsun1997/CoLAKE)
* **KEPLER**: "KEPLER: A Unified Model for Knowledge Embedding and Pre-trained Language Representation". TACL(2020) [[PDF]](https://arxiv.org/pdf/1911.06136.pdf) [[code]](https://github.com/THU-KEG/KEPLER)
* **LUKE**: "LUKE: Deep Contextualized Entity Representations with Entity-aware Self-attention". EMNLP(2020) [[PDF]](https://www.aclweb.org/anthology/2020.emnlp-main.523) [[code]](https://github.com/studio-ousia/luke) :star::star::star:
* **GLM**: "Exploiting Structured Knowledge in Text via Graph-Guided Representation Learning". EMNLP(2020) [[PDF]](https://www.aclweb.org/anthology/2020.emnlp-main.722) [[code]](https://github.com/taoshen58/glm-codes)
* **GRF**: "Language Generation with Multi-Hop Reasoning on Commonsense Knowledge Graph". EMNLP(2020) [[PDF]](https://arxiv.org/abs/2009.11692) [[code]](https://github.com/cdjhz/multigen)
* **K-BERT**: "K-BERT: Enabling Language Representation with Knowledge Graph". AAAI(2020) [[PDF]](https://mx.aaai.org/ojs/index.php/AAAI/article/view/5681) [[code]](https://github.com/autoliuweijie/K-BERT) :star::star::star:
* **LM-as-KG**: "Language Models are Open Knowledge Graphs". arXiv(2020) [[PDF]](https://arxiv.org/abs/2010.11967)
* **LAMA**: "Language Models as Knowledge Bases?". EMNLP(2019) [[PDF]](https://arxiv.org/abs/1909.01066) [[code]](https://github.com/facebookresearch/LAMA) :star::star::star:
* **COMET**: "COMET: Commonsense Transformers for Automatic Knowledge Graph Construction". ACL(2019) [[PDF]](https://arxiv.org/abs/1906.05317) [[code]](https://github.com/atcbosselut/comet-commonsense) :star::star::star:
* **ERNIE(Tsinghua)**: "ERNIE: Enhanced Language Representation with Informative Entities". ACL(2019) [[PDF]](https://www.aclweb.org/anthology/P19-1139.pdf) [[code]](https://github.com/thunlp/ERNIE)
* **ERNIE(Baidu)**: "ERNIE: Enhanced Representation through Knowledge Integration". arXiv(2019) [[PDF]](https://arxiv.org/pdf/1904.09223.pdf) [[code]](https://github.com/PaddlePaddle/ERNIE)


## Dialogue System

### Survey
* **Survey of Dialogue**: "Recent Advances in Deep Learning Based Dialogue Systems: A Systematic Survey". arXiv(2021) [[PDF]](https://arxiv.org/abs/2105.04387) :star::star::star::star:
* **Survey of Open-domain Dialogue**: "Challenges in Building Intelligent Open-domain Dialog Systems". TOIS(2020) [[PDF]](https://dl.acm.org/doi/10.1145/3383123) :star::star::star::star:
* **Survey of Dialogue**: "A Survey on Dialogue Systems: Recent Advances and New Frontiers". SIGKDD Explorations(2017) [[PDF]](https://arxiv.org/pdf/1711.01731.pdf)
* **Survey of Corpora**: "A Survey of Available Corpora For Building Data-Driven Dialogue Systems". arXiv(2017) [[PDF]](https://arxiv.org/pdf/1512.05742.pdf) [[data]](https://breakend.github.io/DialogDatasets/)


### Target-guided Dialogue
* **CG-nAR**: "Thinking Clearly, Talking Fast: Concept-Guided Non-Autoregressive Generation for Open-Domain Dialogue Systems". EMNLP(2021) [[PDF]](https://arxiv.org/abs/2109.04084) [[code]](https://github.com/RowitZou/CG-nAR) :star::star::star:
* **DiSCoL**: "DiSCoL: Toward Engaging Dialogue Systems through Conversational Line Guided Response Generation". NAACL(2021) [[PDF]](https://www.aclweb.org/anthology/2021.naacl-demos.4) [[code]](https://github.com/PlusLabNLP/Dialogue_System_Hackathon)
* **OTTers**: "OTTers: One-turn Topic Transitions for Open-Domain Dialogue". ACL(2021) [[PDF]](https://arxiv.org/abs/2105.13710) [[data]](https://github.com/karinseve/OTTers)
* **DialoGraph**: "DialoGraph: Incorporating Interpretable Strategy-Graph Networks into Negotiation Dialogues". ICLR(2021) [[PDF]](https://arxiv.org/abs/2106.00920) [[code]](https://github.com/rishabhjoshi/DialoGraph_ICLR21) :star::star::star:
* **FeHED**: "Augmenting Non-Collaborative Dialog Systems with Explicit Semantic and Strategic Dialog History". ICLR(2020) [[PDF]](https://openreview.net/forum?id=ryxQuANKPB) [[code]](https://github.com/zhouyiheng11/augmenting-non-collabrative-dialog)
* **TG-ReDial**: "Towards Topic-Guided Conversational Recommender System". COLING(2020) [[PDF]](https://www.aclweb.org/anthology/2020.coling-main.365.pdf) [[code]](https://github.com/RUCAIBox/TG-ReDial)
* **CG-Policy**: "Conversational Graph Grounded Policy Learning for Open-Domain Conversation Generation". ACL(2020) [[PDF]](https://www.aclweb.org/anthology/2020.acl-main.166)
* **PersuasionForGood**: "Persuasion for Good: Towards a Personalized Persuasive Dialogue System for Social Good". ACL(2019) [[PDF]](https://aclanthology.org/P19-1566) [[data]](https://gitlab.com/ucdavisnlp/persuasionforgood)
* **DuConv**: "Proactive Human-Machine Conversation with Explicit Conversation Goals". ACL(2019) [[PDF]](https://www.aclweb.org/anthology/P19-1369) [[code]](https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2019-DuConv)
* **CKC**: "Keyword-Guided Neural Conversational Model". AAAI(2021) [[PDF]](https://arxiv.org/abs/2012.08383)
* **KnowHRL**: "Knowledge Graph Grounded Goal Planning for Open-Domain Conversation Generation". AAAI(2020) [[PDF]](https://aaai.org/ojs/index.php/AAAI/article/view/6474)
* **DKRN**: "Dynamic Knowledge Routing Network For Target-Guided Open-Domain Conversation". AAAI(2020) [[PDF]](https://arxiv.org/abs/2002.01196)
* **TGConv**: "Target-Guided Open-Domain Conversation". ACL(2019) [[PDF]](https://aclanthology.org/P19-1565/) [[code]](https://github.com/squareRoot3/Target-Guided-Conversation)


### Recommendation Dialogue and CRS
* **KERS**: "KERS: A Knowledge-Enhanced Framework for Recommendation Dialog Systems with Multiple Subgoals". EMNLP-Findings(2021) [[PDF]](https://aclanthology.org/2021.findings-emnlp.94) [[code]](https://github.com/z562/KERS)
* **DuRecDial2.0**: "DuRecDial 2.0: A Bilingual Parallel Corpus for Conversational Recommendation". EMNLP(2021) [[PDF]](https://arxiv.org/abs/2109.08877) [[code]](https://github.com/liuzeming01/DuRecDial)
* **DuRecDial**: "Towards Conversational Recommendation over Multi-Type Dialogs". ACL(2020) [[PDF]](https://arxiv.org/pdf/2005.03954.pdf) [[code]](https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2020-DuRecDial) :star::star::star::star:
* **INSPIRED**: "INSPIRED: Toward Sociable Recommendation Dialog Systems". EMNLP(2020) [[PDF]](https://www.aclweb.org/anthology/2020.emnlp-main.654.pdf) [[data]](https://github.com/sweetpeach/Inspired)
* **GoRecDial**: "Recommendation as a Communication Game: Self-Supervised Bot-Play for Goal-oriented Dialogue". EMNLP(2019) [[PDF]](https://www.aclweb.org/anthology/D19-1203.pdf) [[code]](https://github.com/facebookresearch/ParlAI)
* **CRS-Survey**: "A Survey on Conversational Recommender Systems". ACM Computing Surveys(2021) [[PDF]](https://arxiv.org/abs/2004.00646)
* **CRS-Survey**: "Advances and Challenges in Conversational Recommender Systems: A Survey
". arXiv(2021) [[PDF]](https://arxiv.org/abs/2101.09459)
* **CRSLab**: "CRSLab: An Open-Source Toolkit for Building Conversational Recommender System". arXiv(2021) [[PDF]](https://arxiv.org/pdf/2101.00939.pdf) [[code]](https://github.com/RUCAIBox/CRSLab) :star::star::star:
* **C2-CRS**: "C2-CRS: Coarse-to-Fine Contrastive Learning for Conversational Recommender System". WSDM(2022) [[PDF]](https://arxiv.org/abs/2201.02732) [[code]](https://github.com/RUCAIBox/WSDM2022-C2CRS)
* **MESE**: "Improving Conversational Recommendation Systems' Quality with Context-Aware Item Meta Information". arXiv(2021) [[PDF]](https://arxiv.org/abs/2112.08140) [[code]](https://github.com/by2299/MESE)
* **BotPlay**: "Self-Supervised Bot Play for Conversational Recommendation with Justifications". arXiv(2021) [[PDF]](https://arxiv.org/abs/2112.05197)
* **RID**: "Finetuning Large-Scale Pre-trained Language Models for Conversational Recommendation with Knowledge Graph". arXiv(2021) [[PDF]](https://arxiv.org/abs/2110.07477) [[code]](https://github.com/Lingzhi-WANG/PLM-BasedCRS)
* **CRFR**: "CRFR: Improving Conversational Recommender Systems via Flexible Fragments Reasoning on Knowledge Graphs". EMNLP(2021) [[PDF]](https://aclanthology.org/2021.emnlp-main.355) :star::star::star:
* **NTRD**: "Learning Neural Templates for Recommender Dialogue System". EMNLP(2021) [[PDF]](https://arxiv.org/abs/2109.12302) [[code]](https://github.com/jokieleung/NTRD)
* **CR-Walker**: "CR-Walker: Tree-Structured Graph Reasoning and Dialog Acts for Conversational Recommendation". EMNLP(2021) [[PDF]](https://arxiv.org/abs/2010.10333) [[code]](https://github.com/truthless11/CR-Walker) :star::star::star::star:
* **RevCore**: "RevCore: Review-augmented Conversational Recommendation". ACL-Findings(2021) [[PDF]](https://arxiv.org/abs/2106.00957) [[code]](https://github.com/JD-AI-Research-NLP/RevCore)
* **KECRS**: "KECRS: Towards Knowledge-Enriched Conversational Recommendation System". arXiv(2021) [[PDF]](https://arxiv.org/abs/2105.08261)
* **FPAN**: "Adapting User Preference to Online Feedback in Multi-round Conversational Recommendation". WSDM(2021) [[PDF]](https://dl.acm.org/doi/10.1145/3437963.3441791) [[code]](https://github.com/xxkkrr/FPAN)
* **UNICORN**: "Unified Conversational Recommendation Policy Learning via Graph-based Reinforcement Learning". SIGIR(2021) [[PDF]](https://arxiv.org/abs/2105.09710) [[code]](https://github.com/dengyang17/unicorn)
* **KGSF**: "Improving Conversational Recommender Systems via Knowledge Graph based Semantic Fusion". KDD(2020) [[PDF]](https://arxiv.org/pdf/2007.04032.pdf) [[code]](https://github.com/RUCAIBox/KGSF)
* **CPR**: "Interactive Path Reasoning on Graph for Conversational Recommendation". KDD(2020) [[PDF]](https://arxiv.org/abs/2007.00194) [[code]](https://cpr-conv-rec.github.io/)
* **EAR**: "Estimation-Action-Reflection: Towards Deep Interaction Between Conversational and Recommender Systems". WSDM(2020) [[PDF]](https://arxiv.org/abs/2002.09102) [[code]](https://ear-conv-rec.github.io/)
* **KBRD**: "Towards Knowledge-Based Recommender Dialog System". EMNLP(2019) [[PDF]](https://www.aclweb.org/anthology/D19-1189.pdf) [[code]](https://github.com/THUDM/KBRD)
* **ReDial**: "Towards Deep Conversational Recommendations". NeurIPS(2018) [[PDF]](https://papers.nips.cc/paper/8180-towards-deep-conversational-recommendations.pdf) [[data]](https://github.com/ReDialData/website)

### Knowledge-grounded Dialogue
* **PersonaKGC**: "There Are a Thousand Hamlets in a Thousand People's Eyes: Enhancing Knowledge-grounded Dialogue with Personal Memory". ACL(2022) [[PDF]](https://arxiv.org/abs/2204.02624) [[code]](https://github.com/Lucasftc/PersonaKGC)
* **DiffKG**: "Towards Large-Scale Interpretable Knowledge Graph Reasoning for Dialogue Systems". ACL-Findings(2022) [[PDF]](https://arxiv.org/abs/2203.10610) [[code]](https://github.com/Pascalson/DiffKG-Dialog) :star::star::star:
* **KSAM**: "KSAM: Infusing Multi-Source Knowledge into Dialogue Generation via Knowledge Source Aware Multi-Head Decoding". ACL-Findings(2022) [[PDF]](https://aclanthology.org/2022.findings-acl.30)
* **MDSP**: "Multi-Stage Prompting for Knowledgeable Dialogue Generation". ACL-Findings(2022) [[PDF]](https://arxiv.org/abs/2203.08745) [[code]](https://github.com/NVIDIA/Megatron-LM)
* **FSB**: "Few-Shot Bot: Prompt-Based Learning for Dialogue Systems". arXiv(2021) [[PDF]](https://arxiv.org/abs/2110.08118) [[code]](https://github.com/andreamad8/FSB) :star::star::star:
* **P-GDG**: "Exploring Prompt-based Few-shot Learning for Grounded Dialog Generation". arXiv(2021) [[PDF]](https://arxiv.org/abs/2109.06513)
* **KAT-TSLF**: "A Three-Stage Learning Framework for Low-Resource Knowledge-Grounded Dialogue Generation". EMNLP(2021) [[PDF]](https://arxiv.org/abs/2109.04096) [[code]](https://github.com/neukg/KAT-TSLF)
* **DIALKI**: "DIALKI: Knowledge Identification in Conversational Systems through Dialogue-Document Contextualization". EMNLP(2021) [[PDF]](https://aclanthology.org/2021.emnlp-main.140) [[code]](https://github.com/ellenmellon/DIALKI)
* **SKT-KG**: "Augmenting Knowledge-grounded Conversations with Sequential Knowledge Transition". NAACL(2021) [[PDF]](https://www.aclweb.org/anthology/2021.naacl-main.446)
* **MSKE**: "More is Better: Enhancing Open-Domain Dialogue Generation via Multi-Source Heterogeneous Knowledge". EMNLP(2021) [[PDF]](https://aclanthology.org/2021.emnlp-main.175) [[code]](https://github.com/pku-sixing/EMNLP2021-MSKE_Dialog)
* **EARL**: "EARL: Informative Knowledge-Grounded Conversation Generation with Entity-Agnostic Representation Learning". EMNLP(2021) [[PDF]](https://aclanthology.org/2021.emnlp-main.184) [[code]](https://github.com/thu-coai/earl)
* **SECE**: "Space Efficient Context Encoding for Non-Task-Oriented Dialogue Generation with Graph Attention Transformer". ACL(2021) [[PDF]](https://aclanthology.org/2021.acl-long.546) [[code]](https://github.com/fabiangal/space-efficient-context-encoding-acl21) :star::star::star:
* **MIKe**: "Initiative-Aware Self-Supervised Learning for Knowledge-Grounded Conversations". SIGIR(2021) [[PDF]](https://dl.acm.org/doi/10.1145/3404835.3462824) [[code]](https://github.com/ChuanMeng/MIKe)
* **GOKC**: "Learning to Copy Coherent Knowledge for Response Generation". AAAI(2021) [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/17486) [[code]](https://github.com/jq2276/Learning2Copy)
* **KnowledGPT**: "Knowledge-Grounded Dialogue Generation with Pre-trained Language Models". EMNLP(2020) [[PDF]](https://www.aclweb.org/anthology/2020.emnlp-main.272) [[code]](https://github.com/zhaoxlpku/KnowledGPT)
* **DiffKS**: "Difference-aware Knowledge Selection for Knowledge-grounded Conversation Generation". EMNLP-Findings(2020) [[PDF]](https://www.aclweb.org/anthology/2020.findings-emnlp.11) [[code]](https://github.com/chujiezheng/DiffKS)
* **DukeNet**: "DukeNet: A Dual Knowledge Interaction Network for Knowledge-Grounded Conversation". SIGIR(2020) [[PDF]](https://dl.acm.org/doi/10.1145/3397271.3401097) [[code]](https://github.com/ChuanMeng/DukeNet)
* **CCN**: "Cross Copy Network for Dialogue Generation". EMNLP(2020) [[PDF]](https://www.aclweb.org/anthology/2020.emnlp-main.149) [[code]](https://github.com/jichangzhen/CCN)
* **PIPM**: "Bridging the Gap between Prior and Posterior Knowledge Selection for Knowledge-Grounded Dialogue Generation". EMNLP(2020) [[PDF]](https://www.aclweb.org/anthology/2020.emnlp-main.275)
* **ConceptFlow**: "Grounded Conversation Generation as Guided Traverses in Commonsense Knowledge Graphs". ACL(2020) [[PDF]](https://aclanthology.org/2020.acl-main.184/) [[code]](https://github.com/thunlp/ConceptFlow) :star::star::star::star:
* **ConKADI**: "Diverse and Informative Dialogue Generation with Context-Specific Commonsense Knowledge Awareness". ACL(2020) [[PDF]](https://www.aclweb.org/anthology/2020.acl-main.515) [[code]](https://github.com/pku-sixing/ACL2020-ConKADI) :star::star::star:
* **KIC**: "Generating Informative Conversational Response using Recurrent Knowledge-Interaction and Knowledge-Copy". ACL(2020) [[PDF]](https://www.aclweb.org/anthology/2020.acl-main.6)
* **SKT**: "Sequential Latent Knowledge Selection for Knowledge-Grounded Dialogue". ICLR(2020) [[PDF]](https://openreview.net/pdf?id=Hke0K1HKwr) [[code]](https://github.com/bckim92/sequential-knowledge-transformer) :star::star::star:
* **KdConv**: "KdConv: A Chinese Multi-domain Dialogue Dataset Towards Multi-turn Knowledge-driven Conversation". ACL(2020) [[PDF]](https://arxiv.org/pdf/2004.04100.pdf) [[data]](https://github.com/thu-coai/KdConv)
* **TransDG**: "Improving Knowledge-aware Dialogue Generation via Knowledge Base Question Answering". AAAI(2020) [[PDF]](https://arxiv.org/abs/1912.07491) [[code]](https://github.com/siat-nlp/TransDG)
* **RefNet**: "RefNet: A Reference-aware Network for Background Based Conversation". AAAI(2020) [[PDF]](https://arxiv.org/pdf/1908.06449.pdf) [[code]](https://github.com/ChuanMeng/RefNet)
* **GLKS**: "Thinking Globally, Acting Locally: Distantly Supervised Global-to-Local Knowledge Selection for Background Based Conversation". AAAI(2020) [[PDF]](https://arxiv.org/pdf/1908.09528.pdf) [[code]](https://github.com/PengjieRen/GLKS)
* **AKGCM**: "Knowledge Aware Conversation Generation with Explainable Reasoning over Augmented Graphs". EMNLP(2019) [[PDF]](https://aclanthology.org/D19-1187.pdf) [[code]](https://github.com/PaddlePaddle/Research/tree/master/NLP/EMNLP2019-AKGCM)
* **DyKgChat**: "DyKgChat: Benchmarking Dialogue Generation Grounding on Dynamic Knowledge Graphs". EMNLP(2019) [[PDF]](https://aclanthology.org/D19-1194.pdf) [[code]](https://github.com/Pascalson/DyKGChat)
* **OpenDialKG**: "OpenDialKG: Explainable Conversational Reasoning with Attention-based Walks over Knowledge Graphs". ACL(2019) [[PDF]](https://www.aclweb.org/anthology/P19-1081) [[data]](https://github.com/facebookresearch/opendialkg)
* **WoW**: "Wizard of Wikipedia: Knowledge-Powered Conversational agents". ICLR(2019) [[PDF]](https://arxiv.org/pdf/1811.01241.pdf)
* **PostKS**: "Learning to Select Knowledge for Response Generation in Dialog Systems". IJCAI(2019) [[PDF]](https://www.ijcai.org/proceedings/2019/0706.pdf) [[code-1]](https://github.com/siat-nlp/dialogue-models/tree/master/PostKS) [[code-2]](https://github.com/bzantium/Posterior-Knowledge-Selection) :star::star::star:
* **NKD**: "Knowledge Diffusion for Neural Dialogue Generation". ACL(2018) [[PDF]](https://www.aclweb.org/anthology/P18-1138) [[data]](https://github.com/liushuman/neural-knowledge-diffusion) 
* **Dual Fusion**: "Smarter Response with Proactive Suggestion: A New Generative Neural Conversation Paradigm". IJCAI(2018) [[PDF]](https://www.ijcai.org/proceedings/2018/0629.pdf)
* **CCM**: "Commonsense Knowledge Aware Conversation Generation with Graph Attention". IJCAI(2018) [[PDF]](https://www.ijcai.org/proceedings/2018/0643.pdf) [[code-tf]](https://github.com/tuxchow/ccm) [[code-py]](https://github.com/Lyusungwon/CCM-pytorch)  :star::star::star::star::star:
* **MTask**: "A Knowledge-Grounded Neural Conversation Model". AAAI(2018)  [[PDF]](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16710/16057)
* **GenDS**: "Flexible End-to-End Dialogue System for Knowledge Grounded Conversation". arXiv(2017) [[PDF]](https://arxiv.org/pdf/1709.04264.pdf)


### Task-oriented Dialogue
* **NS-Dial**: "An Interpretable Neuro-Symbolic Reasoning Framework for Task-Oriented Dialogue Generation". ACL(2022) [[PDF]](https://arxiv.org/abs/2203.05843) [[code]](https://github.com/shiquanyang/NS-Dial)
* **GALAXY**: "GALAXY: A Generative Pre-trained Model for Task-Oriented Dialog with Semi-Supervised Learning and Explicit Policy Injection". AAAI(2022) [[PDF]](https://arxiv.org/abs/2111.14592) [[code]](https://github.com/siat-nlp/GALAXY)
* **PPTOD**: "Multi-Task Pre-Training for Plug-and-Play Task-Oriented Dialogue System". arXiv(2021) [[PDF]](https://arxiv.org/abs/2109.14739) [[code]](https://github.com/awslabs/pptod)
* **ToDCL**: "Continual Learning in Task-Oriented Dialogue Systems". EMNLP(2021) [[PDF]](https://aclanthology.org/2021.emnlp-main.590) [[code]](https://github.com/andreamad8/ToDCL)
* **IR-Net**: "Intention Reasoning Network for Multi-Domain End-to-end Task-Oriented Dialogue". EMNLP(2021) [[PDF]](https://aclanthology.org/2021.emnlp-main.174)
* **HyKnow**: "HyKnow: End-to-End Task-Oriented Dialog Modeling with Hybrid Knowledge Management". ACL-Findings(2021) [[PDF]](https://arxiv.org/abs/2105.06041) [[code]](https://github.com/truthless11/HyKnow)
* **DDMN**: "Dual Dynamic Memory Network for End-to-End Multi-turn Task-oriented Dialog Systems". COLING(2020) [[PDF]](https://www.aclweb.org/anthology/2020.coling-main.362/) [[code]](https://github.com/siat-nlp/DDMN)
* **ToD-BERT**: "ToD-BERT: Pre-trained Natural Language Understanding for Task-Oriented Dialogues". EMNLP(2020) [[PDF]](https://www.aclweb.org/anthology/2020.emnlp-main.66) [[code]](https://github.com/jasonwu0731/ToD-BERT)
* **GraphDialog**: "GraphDialog: Integrating Graph Knowledge into End-to-End Task-Oriented Dialogue Systems". EMNLP(2020) [[PDF]](https://www.aclweb.org/anthology/2020.emnlp-main.147) [[code]](https://github.com/shiquanyang/GraphDialog)
* **MARCO**: "Multi-Domain Dialogue Acts and Response Co-Generation". ACL(2020) [[PDF]](https://arxiv.org/abs/2004.12363) [[code]](https://github.com/InitialBug/MarCo-Dialog) :star::star::star:
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
* **SDN**: "Subgoal Discovery for Hierarchical Dialogue Policy Learning". EMNLP(2018) [[PDF]](https://arxiv.org/abs/1804.07855) :star::star::star:
* **D3Q**: "Discriminative Deep Dyna-Q: Robust Planning for Dialogue Policy Learning". EMNLP(2018) [[PDF]](https://arxiv.org/abs/1808.09442) [[code]](https://github.com/MiuLab/D3Q)
* **DDQ**: "Deep Dyna-Q: Integrating Planning for Task-Completion Dialogue Policy Learning". ACL(2018) [[PDF]](https://aclweb.org/anthology/P18-1203) [[code]](https://github.com/MiuLab/DDQ)
* **MAD**: "Memory-augmented Dialogue Management for Task-oriented Dialogue Systems". TOIS(2018) [[PDF]](https://arxiv.org/pdf/1805.00150.pdf)
* **TSCP**: "Sequicity: Simplifying Task-oriented Dialogue Systems with Single Sequence-to-Sequence Architectures". ACL(2018) [[PDF]](https://www.aclweb.org/anthology/P18-1133) [[code]](https://github.com/WING-NUS/sequicity)
* **Mem2Seq**: "Mem2Seq: Effectively Incorporating Knowledge Bases into End-to-End Task-Oriented Dialog Systems". ACL(2018) [[PDF]](https://www.aclweb.org/anthology/P18-1136) [[code]](https://github.com/HLTCHKUST/Mem2Seq) :star::star::star::star:
* **Topic-Seg-Label**: "A Weakly Supervised Method for Topic Segmentation and Labeling in Goal-oriented Dialogues via Reinforcement Learning". IJCAI(2018) [[PDF]](https://www.ijcai.org/proceedings/2018/0612.pdf) [[code]](https://github.com/truthless11/Topic-Seg-Label)
* **AliMe**: "AliMe Chat: A Sequence to Sequence and Rerank based Chatbot Engine". ACL(2017) [[PDF]](https://aclweb.org/anthology/P17-2079)
* **KVR Net**: "Key-Value Retrieval Networks for Task-Oriented Dialogue". SIGDIAL(2017) [[PDF]](https://www.aclweb.org/anthology/W17-5506) [[data]](https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/)


### Emotion-aware Dialogue
* **MISC**: "MISC: A MIxed Strategy-Aware Model Integrating COMET for Emotional Support Conversation". ACL(2022) [[PDF]](https://arxiv.org/abs/2203.13560) [[code]](https://github.com/morecry/MISC)
* **C3KG**: "C3KG: A Chinese Commonsense Conversation Knowledge Graph". ACL-Findings(2022) [[PDF]](https://arxiv.org/abs/2204.02549) [[data]](https://github.com/XiaoMi/C3KG)
* **GEE**: "Perspective-taking and Pragmatics for Generating Empathetic Responses Focused on Emotion Causes". EMNLP(2021) [[PDF]](https://arxiv.org/abs/2109.08828) [[code]](https://github.com/skywalker023/focused-empathy)
* **RecEC**: "Improving Empathetic Response Generation by Recognizing Emotion Cause in Conversations". EMNLP-Findings(2021) [[PDF]](https://aclanthology.org/2021.findings-emnlp.70) [[code]](https://github.com/A-Rain/EmpDialogue_RecEC)
* **CoMAE**: "CoMAE: A Multi-factor Hierarchical Framework for Empathetic Response Generation". ACL-Findings(2021) [[PDF]](https://aclanthology.org/2021.findings-acl.72) [[code]](https://github.com/chujiezheng/CoMAE)
* **DAG-ERC**: "Directed Acyclic Graph Network for Conversational Emotion Recognition". ACL(2021) [[PDF]](https://arxiv.org/abs/2105.12907) [[code]](https://github.com/shenwzh3/DAG-ERC)
* **DialogueCRN**: "DialogueCRN: Contextual Reasoning Networks for Emotion Recognition in Conversations". ACL(2021) [[PDF]](https://arxiv.org/abs/2106.01978) [[code]](https://github.com/zerohd4869/DialogueCRN)
* **TodKAT**: "Topic-Driven and Knowledge-Aware Transformer for Dialogue Emotion Detection". ACL(2021) [[PDF]](https://arxiv.org/abs/2106.01071) [[code]](https://github.com/something678/TodKat)
* **ESConv**: "Towards Emotional Support Dialog Systems". ACL(2021) [[PDF]](https://arxiv.org/abs/2106.01144) [[data]](https://github.com/thu-coai/Emotional-Support-Conversation) :star::star::star:
* **EmpDG**: "EmpDG: Multi-resolution Interactive Empathetic Dialogue Generation". COLING(2020) [[PDF]](https://aclanthology.org/2020.coling-main.394) [[code]](https://github.com/qtli/EmpDG)
* **MIME**: "MIME: MIMicking Emotions for Empathetic Response Generation". EMNLP(2020) [[PDF]](https://arxiv.org/abs/2010.01454) [[code]](https://github.com/declare-lab/MIME)
* **PEC**: "Towards Persona-Based Empathetic Conversational Models". EMNLP(2020) [[PDF]](https://aclanthology.org/2020.emnlp-main.531) [[code]](https://github.com/zhongpeixiang/PEC)
* **MoEL**: "MoEL: Mixture of Empathetic Listeners". EMNLP(2019) [[PDF]](https://aclanthology.org/D19-1012) [[code]](https://github.com/HLTCHKUST/MoEL)
* **EmpatheticDialogues**: "Towards Empathetic Open-domain Conversation Models: A New Benchmark and Dataset". ACL(2019) [[PDF]](https://aclanthology.org/P19-1534) [[data]](https://github.com/facebookresearch/EmpatheticDialogues) :star::star::star:
* **EmoDS**: "Generating Responses with a Specific Emotion in Dialog". ACL(2019) [[PDF]](https://aclanthology.org/P19-1359)
* **MojiTalk**: "MojiTalk: Generating Emotional Responses at Scale". ACL(2018) [[PDF]](https://aclanthology.org/P18-1104)
* **ECM**: "Emotional Chatting Machine: Emotional Conversation Generation with Internal and External Memory". AAAI(2018) [[PDF]](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16455/15753) [[code]](https://github.com/tuxchow/ecm)


### Persona-grounded Dialogue
* **DuLeMon**: "Long Time No See! Open-Domain Conversation with Long-Term Persona Memory". ACL-Findings(2022) [[PDF]](https://arxiv.org/abs/2203.05797) [[data]](https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2022-DuLeMon) :star::star::star:
* **GME**: "Transferable Persona-Grounded Dialogues via Grounded Minimal Edits". EMNLP(2021) [[PDF]](https://arxiv.org/abs/2109.07713) [[code]](https://github.com/thu-coai/grounded-minimal-edit)
* **BoB**: "BoB: BERT Over BERT for Training Persona-based Dialogue Models from Limited Personalized Data". ACL(2021) [[PDF]](https://aclanthology.org/2021.acl-long.14) [[code]](https://github.com/songhaoyu/BoB)
* **PABST**: "Unsupervised Enrichment of Persona-grounded Dialog with Background Stories". ACL(2021) [[PDF]](https://arxiv.org/abs/2106.08364) [[code]](https://github.com/majumderb/pabst)
* **pragmatic-consistency**: "Will I Sound Like Me? Improving Persona Consistency in Dialogues through Pragmatic Self-Consciousness". EMNLP(2020) [[PDF]](https://arxiv.org/abs/2004.05816) [[code]](https://github.com/skywalker023/pragmatic-consistency) :star::star::star::star:
* **XPersona**: "XPersona: Evaluating Multilingual Personalized Chatbot". arXiv(2020) [[PDF]](https://arxiv.org/abs/2003.07568) [[data]](https://github.com/HLTCHKUST/Xpersona)
* **P^2Bot**: "You Impress Me: Dialogue Generation via Mutual Persona Perception". ACL(2020) [[PDF]](https://aclanthology.org/2020.acl-main.131) [[code]](https://github.com/SivilTaram/Persona-Dialogue-Generation)
* **PAML**: "Personalizing Dialogue Agents via Meta-Learning". ACL(2019) [[PDF]](https://www.aclweb.org/anthology/P19-1542) [[code]](https://github.com/HLTCHKUST/PAML)
* **PersonalDilaog**: "Personalized Dialogue Generation with Diversified Traits". arXiv(2019) [[PDF]](https://arxiv.org/abs/1901.09672) [[data]](https://github.com/silverriver/PersonalDilaog)
* **PersonaChat**: "Personalizing Dialogue Agents: I have a dog, do you have pets too?" ACL(2018) [[PDF]](https://aclanthology.org/P18-1205) [[data]](https://github.com/facebookresearch/ParlAI/tree/main/projects/personachat) :star::star::star::star:
* **PCCM**: "Assigning Personality/Profile to a Chatting Machine for Coherent Conversation Generation". IJCAI(2018) [[PDF]](https://www.ijcai.org/proceedings/2018/0595.pdf)


### Open-domain Dialogue
* **Overview**: "Open-domain Dialogue Generation: What We Can Do, Cannot Do, And Should Do Next". ACL-NLP4ConvAI(2022) [[PDF]](https://aclanthology.org/2022.nlp4convai-1.13)
* **ProphetChat**: "ProphetChat: Enhancing Dialogue Generation with Simulation of Future Conversation". ACL(2022) [[PDF]](https://aclanthology.org/2022.acl-long.68)  :star::star::star:
* **DialoFlow**: "Conversations Are Not Flat: Modeling the Dynamic Information Flow across Dialogue Utterances". ACL(2021) [[PDF]](https://arxiv.org/abs/2106.02227) [[code]](https://github.com/ictnlp/DialoFlow)
* **DialogBERT**: "DialogBERT: Discourse-Aware Response Generation via Learning to Recover and Rank Utterances". AAAI(2021) [[PDF]](https://arxiv.org/pdf/2012.01775.pdf)
* **CDial-GPT**: "A Large-Scale Chinese Short-Text Conversation Dataset". NLPCC(2020) [[PDF]](https://arxiv.org/pdf/2008.03946.pdf) [[code]](https://github.com/thu-coai/CDial-GPT)
* **DialoGPT**: "DialoGPT : Large-Scale Generative Pre-training for Conversational Response Generation". ACL(2020) [[PDF]](https://arxiv.org/pdf/1911.00536.pdf) [[code]](https://github.com/microsoft/DialoGPT) :star::star::star:
* **PLATO-XL**: "PLATO-XL: Exploring the Large-scale Pre-training of Dialogue Generation". arXiv(2021) [[PDF]](https://arxiv.org/abs/2109.09519) [[code]](https://github.com/PaddlePaddle/Knover/tree/develop/projects)
* **PLATO-2**: "PLATO-2: Towards Building an Open-Domain Chatbot via Curriculum Learning". ACL-Findings(2021) [[PDF]](https://arxiv.org/abs/2006.16779) [[code]](https://github.com/PaddlePaddle/Knover/tree/develop/projects/PLATO-2)
* **PLATO**: "PLATO: Pre-trained Dialogue Generation Model with Discrete Latent Variable". ACL(2020) [[PDF]](https://arxiv.org/pdf/1910.07931.pdf) [[code]](https://github.com/PaddlePaddle/Research/tree/master/NLP/Dialogue-PLATO) 
* **Guyu**: "An Empirical Investigation of Pre-Trained Transformer Language Models for Open-Domain Dialogue Generation". arXiv(2020) [[PDF]](https://arxiv.org/pdf/2003.04195.pdf) [[code]](https://github.com/lipiji/Guyu)
* **CL4Dialogue**: "Group-wise Contrastive Learning for Neural Dialogue Generation". EMNLP-Findings(2020) [[PDF]](https://www.aclweb.org/anthology/2020.findings-emnlp.70) [[code]](https://github.com/hengyicai/ContrastiveLearning4Dialogue) :star::star::star:
* **Neg-train**: "Negative Training for Neural Dialogue Response Generation". ACL(2020) [[PDF]](https://www.aclweb.org/anthology/2020.acl-main.185) [[code]](https://github.mit.edu/tianxing/negativetraining_acl2020)
* **HDSA**: "Semantically Conditioned Dialog Response Generation via Hierarchical Disentangled Self-Attention". ACL(2019) [[PDF]](https://www.aclweb.org/anthology/P19-1360) [[code]](https://github.com/wenhuchen/HDSA-Dialog) :star::star::star:
* **CAS**: "Skeleton-to-Response: Dialogue Generation Guided by Retrieval Memory". NAACL(2019) [[PDF]](https://www.aclweb.org/anthology/N19-1124) [[code]](https://github.com/jcyk/Skeleton-to-Response)
* **Edit-N-Rerank**: "Response Generation by Context-aware Prototype Editing". AAAI(2019) [[PDF]](https://arxiv.org/pdf/1806.07042.pdf) [[code]](https://github.com/MarkWuNLP/ResponseEdit) :star::star::star:
* **HVMN**: "Hierarchical Variational Memory Network for Dialogue Generation". WWW(2018) [[PDF]](https://dl.acm.org/citation.cfm?doid=3178876.3186077) [[code]](https://github.com/chenhongshen/HVMN)
* **XiaoIce**: "The Design and Implementation of XiaoIce, an Empathetic Social Chatbot". arXiv(2018) [[PDF]](https://arxiv.org/pdf/1812.08989.pdf) :star::star::star:
* **D2A**: "Dialog-to-Action: Conversational Question Answering Over a Large-Scale Knowledge Base". NeurIPS(2018) [[PDF]](https://papers.nips.cc/paper/7558-dialog-to-action-conversational-question-answering-over-a-large-scale-knowledge-base.pdf) [[code]](https://github.com/guoday/Dialog-to-Action)
* **DAIM**: "Generating Informative and Diverse Conversational Responses via Adversarial Information Maximization". NeurIPS(2018) [[PDF]](https://papers.nips.cc/paper/7452-generating-informative-and-diverse-conversational-responses-via-adversarial-information-maximization.pdf)
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
* **MMI**: "A Diversity-Promoting Objective Function for Neural Conversation Models". NAACL(2016)  [[PDF]](https://www.aclweb.org/anthology/N16-1014) [[code]](https://github.com/jiweil/Neural-Dialogue-Generation)


### Dialogue Evaluation
* **RoMe**: "RoMe: A Robust Metric for Evaluating Natural Language Generation". ACL(2022) [[PDF]](https://arxiv.org/abs/2203.09183) [[code]](https://github.com/rashad101/RoMe) 
* **CTC-Score**: "Compression, Transduction, and Creation: A Unified Framework for Evaluating Natural Language Generation". EMNLP(2021) [[PDF]](https://arxiv.org/abs/2109.06379) [[code]](https://github.com/tanyuqian/ctc-gen-eval)
* **QuantiDCE**: "Towards Quantifiable Dialogue Coherence Evaluation". ACL(2021) [[PDF]](https://arxiv.org/abs/2106.00507) [[code]](https://github.com/James-Yip/QuantiDCE)
* **DynaEval**: "DynaEval: Unifying Turn and Dialogue Level Evaluation". ACL(2021) [[PDF]](https://aclanthology.org/2021.acl-long.441) [[code]](https://github.com/e0397123/DynaEval)
* **Spot-the-Bot**: "Spot The Bot: A Robust and Efficient Framework for the Evaluation of Conversational Dialogue Systems". EMNLP(2020) [[PDF]](https://www.aclweb.org/anthology/2020.emnlp-main.326) [[code]](https://github.com/jderiu/spot-the-bot-code)
* **BLEURT**: "BLEURT: Learning Robust Metrics for Text Generation". ACL(2020) [[PDF]](https://www.aclweb.org/anthology/2020.acl-main.704) [[code]](https://github.com/google-research/bleurt)
* **Coherence**: "Dialogue Coherence Assessment Without Explicit Dialogue Act Labels". ACL(2020) [[PDF]](https://aclanthology.org/2020.acl-main.133) [[code]](https://github.com/UKPLab/acl2020-dialogue-coherence-assessment)
* **GRADE**: "GRADE: Automatic Graph-Enhanced Coherence Metric for Evaluating Open-Domain Dialogue Systems". ACL(2020) [[PDF]](https://www.aclweb.org/anthology/2020.emnlp-main.742) [[code]](https://github.com/li3cmz/GRADE)
* **uBLEU**: "uBLEU: Uncertainty-Aware Automatic Evaluation Method for Open-Domain Dialogue Systems". ACL(2020) [[PDF]](https://www.aclweb.org/anthology/2020.acl-srw.27) [[code]](https://github.com/YumaTsuta/upsilon_bleu)
* **USR**: "USR: An Unsupervised and Reference Free Evaluation Metric for Dialog Generation". ACL(2020) [[PDF]](https://www.aclweb.org/anthology/2020.acl-main.64) [[code]](https://github.com/Shikib/usr)
* **ADVMT**: "One `Ruler` for All Languages: Multi-Lingual Dialogue Evaluation with Adversarial Multi-Task Learning". IJCAI(2018) [[PDF]](https://www.ijcai.org/proceedings/2018/0616.pdf)



## Natural Language Generation

### Survey for NLG
* **CTG**: "A Survey of Controllable Text Generation using Transformer-based Pre-trained Language Models". arXiv(2022) [[PDF]](https://arxiv.org/abs/2201.05337)
* **RTG**: "A Survey on Retrieval-Augmented Text Generation". arXiv(2022) [[PDF]](https://arxiv.org/abs/2202.01110)
* **Evaluation**: "A Survey of Evaluation Metrics Used for NLG Systems". arXiv(2020) [[PDF]](https://arxiv.org/abs/2008.12009)


### Text Planning
* **Time Control**: "Language Modeling via Stochastic Processes". ICLR(2022) [[PDF]](https://arxiv.org/abs/2203.11370) [[code]](https://github.com/rosewang2008/language_modeling_via_stochastic_processes) :star::star::star::star::star:
* **PLANET**: "PLANET: Dynamic Content Planning in Autoregressive Transformers for Long-form Text Generation". arXiv(2022) [[PDF]](https://arxiv.org/abs/2203.09100)
* **CETP**: "Knowledge-based Review Generation by Coherence Enhanced Text Planning". SIGIR(2021) [[PDF]](https://dl.acm.org/doi/10.1145/3404835.3462865) :star::star::star::star:
* **PlanGen**: "Plan-then-Generate: Controlled Data-to-Text Generation via Planning". EMNLP-Findings(2021) [[PDF]](https://aclanthology.org/2021.findings-emnlp.76) [[code]](https://github.com/yxuansu/PlanGen)
* **DYPLOC**: "DYPLOC: Dynamic Planning of Content Using Mixed Language Models for Text Generation". ACL(2021) [[PDF]](https://arxiv.org/abs/2106.00791) [[code]](https://github.com/XinyuHua/dyploc-acl2021)
* **Tree-PLAN**: "Infobox-to-text Generation with Tree-like Planning based Attention Network". IJCAI(2020) [[PDF]](https://www.ijcai.org/proceedings/2020/522)
* **PAIR**: "PAIR: Planning and Iterative Refinement in Pre-trained Transformers for Long Text Generation". EMNLP(2020) [[PDF]](https://aclanthology.org/2020.emnlp-main.57) [[code]](https://github.com/XinyuHua/pair-emnlp2020)
* **SentPlan**: "Sentence-Level Content Planning and Style Specification for Neural Text Generation". EMNLP(2019) [[PDF]](https://aclanthology.org/D19-1055) [[code]](https://github.com/XinyuHua/textgen-emnlp19)
* **PHVM**: "Long and Diverse Text Generation with Planning-based Hierarchical Variational Model". EMNLP(2019) [[PDF]](https://www.aclweb.org/anthology/D19-1321) [[code]](https://github.com/ZhihongShao/Planning-based-Hierarchical-Variational-Model)


### Language Generation
* **CT-Loss**: "A Simple Contrastive Learning Objective for Alleviating Neural Text Degeneration". arXiv(2022) [[PDF]](https://arxiv.org/abs/2205.02517) [[code]](https://github.com/shaojiejiang/ct-loss)
* **SimCTG**: "A Contrastive Framework for Neural Text Generation". arXiv(2022) [[PDF]](https://arxiv.org/abs/2202.06417) [[code]](https://github.com/yxuansu/simctg) :star::star::star::star:
* **Latent-GLAT**: "*latent*-GLAT: Glancing at Latent Variables for Parallel Text Generation". ACL(2022) [[PDF]](https://arxiv.org/abs/2204.02030) [[code]](https://github.com/baoy-nlp/Latent-GLAT)
* **Lattice Generation**: "Massive-scale Decoding for Text Generation using Lattices". arXiv(2021) [[PDF]](https://arxiv.org/abs/2112.07660) [[code]](https://github.com/jiacheng-xu/lattice-generation)
* **NeuroLogic A*esque**: "NeuroLogic A *esque Decoding: Constrained Text Generation with Lookahead Heuristics". arXiv(2021) [[PDF]](https://arxiv.org/abs/2112.08726)
* **NeuroLogic**: "NeuroLogic Decoding: (Un)supervised Neural Text Generation with Predicate Logic Constraints". NAACL(2021) [[PDF]](https://aclanthology.org/2021.naacl-main.339)
* **s2s-ft**: "s2s-ft: Fine-Tuning Pretrained Transformer Encoders for Sequence-to-Sequence Learning". arXiv(2021) [[PDF]](https://arxiv.org/abs/2110.13640) [[code]](https://github.com/microsoft/unilm/tree/master/s2s-ft)
* **CLAPS**: "Contrastive Learning with Adversarial Perturbations for Conditional Text Generation". ICLR(2021) [[PDF]](https://arxiv.org/abs/2012.07280) [[code]](https://github.com/seanie12/CLAPS) :star::star::star::star:
* **EBM**: "Exposure Bias versus Self-Recovery: Are Distortions Really Incremental for Autoregressive Text Generation?". EMNLP(2021) [[PDF]](https://aclanthology.org/2021.emnlp-main.415)
* **DiscoDVT**: "DiscoDVT: Generating Long Text with Discourse-Aware Discrete Variational Transformer". EMNLP(2021) [[PDF]](https://arxiv.org/abs/2110.05999) [[code]](https://github.com/cdjhz/DiscoDVT)
* **Embedding-Transfer**: "Bridging Subword Gaps in Pretrain-Finetune Paradigm for Natural Language Generation". ACL(2021) [[PDF]](https://arxiv.org/abs/2106.06125) [[code]](https://github.com/DeepLearnXMU/embedding-transfer)
* **FastSeq**: "EL-Attention: Memory Efficient Lossless Attention for Generation". ICML(2021) [[PDF]](https://arxiv.org/abs/2105.04779) [[code]](https://github.com/microsoft/fastseq) :star::star::star:
* **RAG**: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks". NeurIPS(2020) [[PDF]](https://arxiv.org/abs/2005.11401) [[code]](https://github.com/huggingface/transformers/blob/main/examples/research_projects/rag/README.md) :star::star::star::star:
* **Cascaded Generation**: "Cascaded Text Generation with Markov Transformers". NeurIPS(2020) [[PDF]](https://arxiv.org/pdf/2006.01112.pdf) [[code]](https://github.com/harvardnlp/cascaded-generation)
* **POINTER**: "POINTER: Constrained Progressive Text Generation via Insertion-based Generative Pre-training". EMNLP(2020) [[PDF]](https://arxiv.org/abs/2005.00558) [[code]](https://github.com/dreasysnail/POINTER)
* **BERTSeq2Seq**: "Leveraging Pre-trained Checkpoints for Sequence Generation Tasks". TACL(2020) [[PDF]](https://arxiv.org/pdf/1907.12461.pdf) [[code-tf]](https://github.com/google-research/google-research/tree/master/bertseq2seq) [[code-py]](https://github.com/huggingface/transformers) :star::star::star:
* **ERNIE-GEN**: "ERNIE-GEN: An Enhanced Multi-Flow Pre-training and Fine-tuning Framework for Natural Language Generation". IJCAI(2020) [[PDF]](https://arxiv.org/pdf/2001.11314.pdf) [[code]](https://github.com/PaddlePaddle/ERNIE/tree/repro/ernie-gen) :star::star::star:
* **Distill-BERT-Textgen**: "Distilling Knowledge Learned in BERT for Text Generation". ACL(2020) [[PDF]](https://arxiv.org/abs/1911.03829) [[code]](https://github.com/ChenRocks/Distill-BERT-Textgen)
* **Repetition-Problem-NLG**: "A Theoretical Analysis of the Repetition Problem in Text Generation". AAAI(2021) [[PDF]](https://arxiv.org/pdf/2012.14660.pdf) [[code]](https://github.com/fuzihaofzh/repetition-problem-nlg)
* **CoMMA**: "A Study of Non-autoregressive Model for Sequence Generation". ACL(2020) [[PDF]](https://www.aclweb.org/anthology/2020.acl-main.15.pdf)
* **Nucleus Sampling**: "The Curious Case of Neural Text Degeneration". ICLR(2020) [[PDF]](https://openreview.net/forum?id=rygGQyrFvH) [[code]](https://github.com/ari-holtzman/degen) :star::star::star:
* **Entmax**: "Sparse Sequence-to-Sequence Models". ACL(2019) [[PDF]](https://www.aclweb.org/anthology/P19-1146) [[code]](https://github.com/deep-spin/entmax)



## Others

### Text Summarization
* **BERTSum**: "Fine-tune BERT for Extractive Summarization". arXiv(2019) [[PDF]](https://arxiv.org/pdf/1903.10318.pdf) [[code]](https://github.com/nlpyang/BertSum)
* **QASumm**: "Guiding Extractive Summarization with Question-Answering Rewards". NAACL(2019) [[PDF]](https://www.aclweb.org/anthology/N19-1264) [[code]](https://github.com/ucfnlp/summ_qa_rewards)
* **Re^3Sum**: "Retrieve, Rerank and Rewrite: Soft Template Based Neural Summarization". ACL(2018) [[PDF]](https://www.aclweb.org/anthology/P18-1015) [[code]](https://www4.comp.polyu.edu.hk/~cszqcao/data/IRSum_Resource.zip)
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


### Machine Translation
* **VOLT**: "Vocabulary Learning via Optimal Transport for Machine Translation". ACL(2021) [[PDF]](https://arxiv.org/abs/2012.15671) [[code]](https://github.com/Jingjing-NLP/VOLT)
* **OR-NMT**: "Bridging the Gap between Training and Inference for Neural Machine Translation". ACL(2019) [[PDF]](https://aclanthology.org/P19-1426) [[code]](https://github.com/ictnlp/OR-NMT) :star::star::star:
* **Multi-pass decoder**: "Adaptive Multi-pass Decoder for Neural Machine Translation". EMNLP(2018) [[PDF]](https://www.aclweb.org/anthology/D18-1048) 
* **KVMem-Attention**: "Neural Machine Translation with Key-Value Memory-Augmented Attention". IJCAI(2018) [[PDF]](https://www.ijcai.org/proceedings/2018/0357.pdf) :star::star::star:
* **Deliberation Networks**: "Deliberation Networks: Sequence Generation Beyond One-Pass Decoding". NeurIPS(2017) [[PDF]](https://papers.nips.cc/paper/6775-deliberation-networks-sequence-generation-beyond-one-pass-decoding.pdf)
* **Interactive-Attention**: "Interactive Attention for Neural Machine Translation". COLING(2016) [[PDF]](https://www.aclweb.org/anthology/C16-1205)


### Question Answering
* **FlowQA**: "FlowQA: Grasping Flow in History for Conversational Machine Comprehension". ICLR(2019) [[PDF]](https://arxiv.org/pdf/1810.06683.pdf) [[code]](https://github.com/momohuang/FlowQA) :star::star::star::star:
*  **SDNet**: "SDNet: Contextualized Attention-based Deep Network for Conversational Question Answering". arXiv(2018) [[PDF]](https://arxiv.org/pdf/1812.03593.pdf) [[code]](https://github.com/microsoft/SDNet)
* **CFC**: "Coarse-grain Fine-grain Coattention Network for Multi-evidence Question Answering". ICLR(2019) [[PDF]](https://arxiv.org/pdf/1901.00603.pdf)
* **MTQA**: "Multi-Task Learning with Multi-View Attention for Answer Selection and Knowledge Base Question Answering". AAAI(2019) [[PDF]](https://aaai.org/ojs/index.php/AAAI/article/view/4593) [[code]](https://github.com/dengyang17/MTQA)
* **MacNet**: "MacNet: Transferring Knowledge from Machine Comprehension to Sequence-to-Sequence Models". NeurIPS(2018) [[PDF]](https://papers.nips.cc/paper/7848-macnet-transferring-knowledge-from-machine-comprehension-to-sequence-to-sequence-models.pdf)
* **CQG-KBQA**: "Knowledge Base Question Answering via Encoding of Complex Query Graphs". EMNLP(2018) [[PDF]](https://www.aclweb.org/anthology/D18-1242) [[code]](https://202.120.38.146/CompQA/) :star::star::star::star:
* **HR-BiLSTM**: "Improved Neural Relation Detection for Knowledge Base Question Answering". ACL(2017) [[PDF]](https://aclweb.org/anthology/P17-1053)
* **KBQA-CGK**: "An End-to-End Model for Question Answering over Knowledge Base with Cross-Attention Combining Global Knowledge". ACL(2017) [[PDF]](https://aclweb.org/anthology/P17-1021)
* **KVMem**: "Key-Value Memory Networks for Directly Reading Documents". EMNLP(2016) [[PDF]](https://www.aclweb.org/anthology/D16-1147)


### Text Matching
* **Poly-encoder**: "Poly-encoders: Architectures and Pre-training Strategies for Fast and Accurate Multi-sentence Scorings". ICLR(2020) [[PDF]](https://openreview.net/pdf?id=SkxgnnNFvH) [[code]](https://github.com/sfzhou5678/PolyEncoder)
* **AugSBERT**: "Augmented SBERT: Data Augmentation Method for Improving Bi-Encoders for Pairwise Sentence Scoring Tasks". arXiv(2020) [[PDF]](https://arxiv.org/pdf/2010.08240.pdf) [[code]](https://github.com/UKPLab/sentence-transformers)
* **SBERT**: "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks". EMNLP(2019) [[PDF]](https://arxiv.org/pdf/1908.10084.pdf) [[code]](https://github.com/UKPLab/sentence-transformers)