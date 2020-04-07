# Paper-Reading
Paper reading list in natural language processing.


- [Paper-Reading](#paper-reading)
  - [Deep Learning in NLP](#deep-learning-in-nlp)
  - [Dialogue System](#dialogue-system)
    - [Conversational Recommendation](#conversational-recommendation)
    - [Task-oriented Dialogue](#task-oriented-dialogue)
    - [Open-domain Dialogue](#open-domain-dialogue)
    - [Personalized Dialogue](#personalized-dialogue)
    - [Miscellaneous](#miscellaneous)
  - [Knowledge Representation & Reasoning](#knowledge-representation-and-reasoning)
  - [Text Summarization](#text-summarization)
  - [Topic Modeling](#topic-modeling)
  - [Machine Translation](#machine-translation)
  - [Question Answering](#question-answering)
  - [Reading Comprehension](#reading-comprehension)
  - [Image Captioning](#image-captioning)

***

## Deep Learning in NLP
* **Sequence Generation**: "A Generalized Framework of Sequence Generation with Application to Undirected Sequence Models". ICML(2020)(under review) [[PDF]](https://arxiv.org/pdf/1905.12790.pdf) [[code]](https://github.com/nyu-dl/dl4mt-seqgen)
* **Sparse-Seq2Seq**: "Sparse Sequence-to-Sequence Models". ACL(2019) [[PDF]](https://www.aclweb.org/anthology/P19-1146) [[code]](https://github.com/deep-spin/entmax)
* **BERT**: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding". NAACL(2019) [[PDF]](https://www.aclweb.org/anthology/N19-1423) [[code]](https://github.com/google-research/bert)
* **CNM**: "CNM: An Interpretable Complex-valued Network for Matching". NAACL(2019) [[PDF]](https://www.aclweb.org/anthology/N19-1420) [[code]](https://github.com/wabyking/qnn)
* **word2vec**: "word2vec Parameter Learning Explained". arXiv(2016) [[PDF]](https://arxiv.org/pdf/1411.2738.pdf)
* **ELMo**: "Deep contextualized word representations". NAACL(2018) [[PDF]](https://www.aclweb.org/anthology/N18-1202)
* **VAE**: "An Introduction to Variational Autoencoders". arXiv(2019) [[PDF]](https://arxiv.org/pdf/1906.02691.pdf)
* **Transformer**: "Attention is All you Need". NeurIPS(2017) [[PDF]](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) [[code-official]](https://github.com/tensorflow/tensor2tensor) [[code-tf]](https://github.com/Kyubyong/transformer) [[code-py]](https://github.com/jadore801120/attention-is-all-you-need-pytorch)
* **Transformer-XL**: "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context". ACL(2019) [[PDF]](https://www.aclweb.org/anthology/P19-1285) [[code]](https://github.com/kimiyoung/transformer-xl)
* **ConvS2S**: "Convolutional Sequence to Sequence Learning". ICML(2017) [[PDF]](http://proceedings.mlr.press/v70/gehring17a/gehring17a.pdf)
* **Survey on Attention**: "An Introductory Survey on Attention Mechanisms in NLP Problems". arXiv(2018) [[PDF]](https://arxiv.org/pdf/1811.05544.pdf)
* **Additive Attention**: "Neural Machine Translation by Jointly Learning to Align and Translate". ICLR(2015) [[PDF]](https://arxiv.org/pdf/1409.0473.pdf) 
* **Multiplicative Attention**: "Effective Approaches to Attention-based Neural Machine Translation". EMNLP(2015) [[PDF]](https://www.aclweb.org/anthology/D15-1166)
* **Memory Net**: "End-To-End Memory Networks". NeurIPS(2015) [[PDF]](http://papers.nips.cc/paper/5846-end-to-end-memory-networks.pdf)
* **Pointer Net**: "Pointer Networks". NeurIPS(2015) [[PDF]](http://papers.nips.cc/paper/5866-pointer-networks.pdf) 
* **Copying Mechanism**: "Incorporating Copying Mechanism in Sequence-to-Sequence Learning". ACL(2016) [[PDF]](https://www.aclweb.org/anthology/P16-1154)
* **Coverage Mechanism**: "Modeling Coverage for Neural Machine Translation". ACL(2016) [[PDF]](https://www.aclweb.org/anthology/P16-1008)
* **GAN**: "Generative Adversarial Nets". NeurIPS(2014) [[PDF]](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)
* **SeqGAN**: "SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient". AAAI(2017) [[PDF]](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14344/14489) [[code]](https://github.com/LantaoYu/SeqGAN)
* **MacNet**: "MacNet: Transferring Knowledge from Machine
Comprehension to Sequence-to-Sequence Models". NeurIPS(2018) [[PDF]](http://papers.nips.cc/paper/7848-macnet-transferring-knowledge-from-machine-comprehension-to-sequence-to-sequence-models.pdf)
* **Graph2Seq**: "Graph2Seq: Graph to Sequence Learning with Attention-based Neural Networks". arXiv(2018) [[PDF]](https://arxiv.org/pdf/1804.00823.pdf)
* **Pretrained Seq2Seq**: "Unsupervised Pretraining for Sequence to Sequence Learning". EMNLP(2017) [[PDF]](https://www.aclweb.org/anthology/D17-1039)
* **Multi-task Learning**: "An Overview of Multi-Task Learning in Deep Neural Networks". arXiv(2017) [[PDF]](https://arxiv.org/pdf/1706.05098.pdf)
* **Latent Multi-task**: "Latent Multi-task Architecture Learning". AAAI(2019) [[PDF]](https://aaai.org/ojs/index.php/AAAI/article/view/4410) [[code]](https://github.com/sebastianruder/sluice-networks)


## Dialogue System

### Conversational Recommendation
* **DuConv**: "Proactive Human-Machine Conversation with Explicit Conversation Goals". ACL(2019) [[PDF]](https://www.aclweb.org/anthology/P19-1369) [[code]](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/Research/ACL2019-DuConv) :star::star::star::star:
* **KBRD**: "Towards Knowledge-Based Recommender Dialog System". EMNLP(2019) [[PDF]](https://www.aclweb.org/anthology/D19-1189.pdf) [[code]](https://github.com/THUDM/KBRD) :star::star::star::star:
* **ReDial**: "Towards Deep Conversational Recommendations". NeurIPS(2018) [[PDF]](http://papers.nips.cc/paper/8180-towards-deep-conversational-recommendations.pdf) [[data]](https://github.com/ReDialData/website) :star::star:
* **Dual Fusion**: "Smarter Response with Proactive Suggestion: A New Generative Neural Conversation Paradigm". IJCAI(2018) [[PDF]](https://www.ijcai.org/proceedings/2018/0629.pdf) :star::star::star:

### Task-oriented Dialogue
* **MALA**: "MALA: Cross-Domain Dialogue Generation with Action Learning". AAAI(2020) [[PDF]](https://arxiv.org/pdf/1912.08442.pdf) :star::star::star:
* **Task-Oriented Dialogue Systems**: "Learning to Memorize in Neural Task-Oriented Dialogue Systems". HKUST MPhil Thesis(2019) [[PDF]](https://arxiv.org/pdf/1905.07687.pdf) :star::star::star::star:
* **GLMP**: "Global-to-local Memory Pointer Networks for Task-Oriented Dialogue". ICLR(2019) [[PDF]](https://arxiv.org/pdf/1901.04713.pdf) [[code]](https://github.com/jasonwu0731/GLMP) :star::star::star::star:
* **KB Retriever**: "Entity-Consistent End-to-end Task-Oriented Dialogue System with KB Retriever". EMNLP(2019) [[PDF]](https://www.aclweb.org/anthology/D19-1013.pdf) [[data]](https://github.com/yizhen20133868/Retriever-Dialogue) :star::star::star:
* **TRADE**: "Transferable Multi-Domain State Generator for Task-Oriented
Dialogue Systems". ACL(2019) [[PDF]](https://www.aclweb.org/anthology/P19-1078) [[code]](https://github.com/jasonwu0731/trade-dst) :star::star::star::star:
* **WMM2Seq**: "A Working Memory Model for Task-oriented Dialog Response Generation". ACL(2019) [[PDF]](https://www.aclweb.org/anthology/P19-1258) :star::star::
* **Pretrain-Fine-tune**: "Training Neural Response Selection for Task-Oriented Dialogue Systems". ACL(2019) [[PDF]](https://www.aclweb.org/anthology/P19-1536) [[data]](https://github.com/PolyAI-LDN/conversational-datasets) :star::star::star:
* **Multi-level Mem**: "Multi-Level Memory for Task Oriented Dialogs". NAACL(2019) [[PDF]](https://www.aclweb.org/anthology/N19-1375) [[code]](https://github.com/DineshRaghu/multi-level-memory-network)  :star::star::star::star:
* **BossNet**: "Disentangling Language and Knowledge in Task-Oriented Dialogs
". NAACL(2019) [[PDF]](https://www.aclweb.org/anthology/N19-1126) [[code]](https://github.com/dair-iitd/BossNet) :star::star::star:
* **SL+RL**: "Dialogue Learning with Human Teaching and Feedback in End-to-End Trainable Task-Oriented Dialogue Systems". NAACL(2018) [[PDF]](https://www.aclweb.org/anthology/N18-1187) :star::star::star:
* **MAD**: "Memory-augmented Dialogue Management for Task-oriented Dialogue Systems". TOIS(2018) [[PDF]](https://arxiv.org/pdf/1805.00150.pdf) :star::star::star:
* **TSCP**: "Sequicity: Simplifying Task-oriented Dialogue Systems with Single Sequence-to-Sequence Architectures". ACL(2018) [[PDF]](https://www.aclweb.org/anthology/P18-1133) [[code]](https://github.com/WING-NUS/sequicity) :star::star::star:
* **Mem2Seq**: "Mem2Seq: Effectively Incorporating Knowledge Bases into End-to-End Task-Oriented Dialog Systems". ACL(2018) [[PDF]](https://www.aclweb.org/anthology/P18-1136) [[code]](https://github.com/HLTCHKUST/Mem2Seq) :star::star::star::star:
* **DSR**: "Sequence-to-Sequence Learning for Task-oriented Dialogue with Dialogue State Representation". COLING(2018)  [[PDF]](https://www.aclweb.org/anthology/C18-1320) :star::star:
* **StateNet**: "Towards Universal Dialogue State Tracking". EMNLP(2018) [[PDF]](https://www.aclweb.org/anthology/D18-1299) :star:
* **Topic-Seg-Label**: "A Weakly Supervised Method for Topic Segmentation and Labeling in Goal-oriented Dialogues via Reinforcement Learning". IJCAI(2018) [[PDF]](https://www.ijcai.org/proceedings/2018/0612.pdf) [[code]](https://github.com/truthless11/Topic-Seg-Label) :star::star::star::star:
* **AliMe**: "AliMe Chat: A Sequence to Sequence and Rerank based Chatbot Engine". ACL(2017) [[PDF]](https://aclweb.org/anthology/P17-2079) :star:
* **KVR Net**: "Key-Value Retrieval Networks for Task-Oriented Dialogue". SIGDIAL(2017) [[PDF]](https://www.aclweb.org/anthology/W17-5506) [[data]](https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/) :star::star:


### Open-domain Dialogue
* **RefNet**: "RefNet: A Reference-aware Network for Background Based Conversation". AAAI(2020) [[PDF]](https://arxiv.org/pdf/1908.06449.pdf) [[code]](https://github.com/ChuanMeng/RefNet) :star::star::star:
* **GLKS**: "Thinking Globally, Acting Locally: Distantly Supervised Global-to-Local Knowledge Selection for Background Based Conversation". AAAI(2020) [[PDF]](https://arxiv.org/pdf/1908.09528.pdf) [[code]](https://github.com/PengjieRen/GLKS) :star::star::star:
* **HDSA**: "Semantically Conditioned Dialog Response Generation via Hierarchical Disentangled Self-Attention". ACL(2019) [[PDF]](https://www.aclweb.org/anthology/P19-1360) [[code]](https://github.com/wenhuchen/HDSA-Dialog) :star::star::star::star:
* **PostKS**: "Learning to Select Knowledge for Response Generation in Dialog Systems". IJCAI(2019) [[PDF]](https://www.ijcai.org/proceedings/2019/0706.pdf) :star::star:
* **Two-Stage-Transformer**: "Wizard of Wikipedia: Knowledge-Powered Conversational agents". ICLR(2019) [[PDF]](https://arxiv.org/pdf/1811.01241.pdf) :star::star:
* **CAS**: "Skeleton-to-Response: Dialogue Generation Guided by Retrieval Memory". NAACL(2019) [[PDF]](https://www.aclweb.org/anthology/N19-1124) [[code]](https://github.com/jcyk/Skeleton-to-Response) :star::star::star:
* **Edit-N-Rerank**: "Response Generation by Context-aware Prototype Editing". AAAI(2019) [[PDF]](https://arxiv.org/pdf/1806.07042.pdf) [[code]](https://github.com/MarkWuNLP/ResponseEdit) :star::star::star:
* **HVMN**: "Hierarchical Variational Memory Network for Dialogue Generation". WWW(2018) [[PDF]](https://dl.acm.org/citation.cfm?doid=3178876.3186077) [[code]](https://github.com/chenhongshen/HVMN) :star::star::star:
* **XiaoIce**: "The Design and Implementation of XiaoIce, an Empathetic Social Chatbot". arXiv(2018) [[PDF]](https://arxiv.org/pdf/1812.08989.pdf) :star::star::star:
* **D2A**: "Dialog-to-Action: Conversational Question Answering Over a Large-Scale Knowledge Base". NeurIPS(2018) [[PDF]](http://papers.nips.cc/paper/7558-dialog-to-action-conversational-question-answering-over-a-large-scale-knowledge-base.pdf) [[code]](https://github.com/guoday/Dialog-to-Action) :star::star::star:
* **DAIM**: "Generating Informative and Diverse Conversational Responses via Adversarial Information Maximization". NeurIPS(2018) [[PDF]](http://papers.nips.cc/paper/7452-generating-informative-and-diverse-conversational-responses-via-adversarial-information-maximization.pdf) :star::star:
* **MTask**: "A Knowledge-Grounded Neural Conversation Model". AAAI(2018)  [[PDF]](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16710/16057) :star:
* **GenDS**: "Flexible End-to-End Dialogue System for Knowledge Grounded Conversation". arXiv(2017) [[PDF]](https://arxiv.org/pdf/1709.04264.pdf) :star::star:
* **Time-Decay-SLU**: "How Time Matters: Learning Time-Decay Attention for Contextual Spoken Language Understanding in Dialogues". NAACL(2018) [[PDF]](https://www.aclweb.org/anthology/N18-1194) [[code]](https://github.com/MiuLab/Time-Decay-SLU) :star::star::star::star:
* **REASON**: "Dialog Generation Using Multi-turn Reasoning Neural Networks". NAACL(2018) [[PDF]](https://www.aclweb.org/anthology/N18-1186) :star::star::star:
* **STD/HTD**: "Learning to Ask Questions in Open-domain Conversational Systems with Typed Decoders". ACL(2018) [[PDF]](https://www.aclweb.org/anthology/P18-1204) [[code]](https://github.com/victorywys/Learning2Ask_TypedDecoder) :star::star::star:
* **CSF**: "Generating Informative Responses with Controlled Sentence Function". ACL(2018) [[PDF]](https://www.aclweb.org/anthology/P18-1139) [[code]](https://github.com/kepei1106/SentenceFunction) :star::star::star:
* **NKD**: "Knowledge Diffusion for Neural Dialogue Generation". ACL(2018) [[PDF]](https://www.aclweb.org/anthology/P18-1138) [[data]](https://github.com/liushuman/neural-knowledge-diffusion) :star::star:
* **DAWnet**: "Chat More: Deepening and Widening the Chatting Topic via A Deep Model". SIGIR(2018) [[PDF]](https://dl.acm.org/citation.cfm?doid=3209978.3210061) [[code]](https://sigirdawnet.wixsite.com/dawnet) :star::star::star:
* **ZSDG**: "Zero-Shot Dialog Generation with Cross-Domain Latent Actions". SIGDIAL(2018) [[PDF]](https://www.aclweb.org/anthology/W18-5001) [[code]](https://github.com/snakeztc/NeuralDialog-ZSDG) :star::star::star:
* **DUA**: "Modeling Multi-turn Conversation with Deep Utterance Aggregation". COLING(2018) [[PDF]](https://www.aclweb.org/anthology/C18-1317) [[code]](https://github.com/cooelf/DeepUtteranceAggregation) :star::star:
* **Data-Aug**: "Sequence-to-Sequence Data Augmentation for Dialogue Language Understanding". COLING(2018) [[PDF]](https://www.aclweb.org/anthology/C18-1105) [[code]](https://github.com/AtmaHou/Seq2SeqDataAugmentationForLU) :star::star:
* **DC-MMI**: "Generating More Interesting Responses in Neural Conversation Models with Distributional Constraints". EMNLP(2018) [[PDF]](https://www.aclweb.org/anthology/D18-1431) [[code]](https://github.com/abaheti95/DC-NeuralConversation) :star::star:
* **cVAE-XGate/CGate**: "Better Conversations by Modeling, Filtering, and Optimizing for Coherence and Diversity". EMNLP(2018) [[PDF]](https://www.aclweb.org/anthology/D18-1432) [[code]](https://github.com/XinnuoXu/CVAE_Dial) :star::star::star:
* **DAM**: "Multi-Turn Response Selection for Chatbots with Deep Attention
Matching Network". ACL(2018) [[PDF]](https://www.aclweb.org/anthology/P18-1103) [[code]](https://github.com/baidu/Dialogue/tree/master/DAM) :star::star::star::star:
* **SMN**: "Sequential Matching Network: A New Architecture for Multi-turn Response Selection in Retrieval-Based Chatbots". ACL(2017) [[PDF]](https://aclweb.org/anthology/P17-1046)  [[code]](https://github.com/MarkWuNLP/MultiTurnResponseSelection) :star::star::star::star:
* **MMI**: "A Diversity-Promoting Objective Function for Neural Conversation Models". NAACL-HLT(2016)  [[PDF]](https://www.aclweb.org/anthology/N16-1014) [[code]](https://github.com/jiweil/Neural-Dialogue-Generation) :star::star:
* **RL-Dialogue**: "Deep Reinforcement Learning for Dialogue Generation". EMNLP(2016) [[PDF]](https://www.aclweb.org/anthology/D16-1127) :star:
* **TA-Seq2Seq**: "Topic Aware Neural Response Generation". AAAI(2017) [[PDF]](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14563/14260) [[code]](https://github.com/LynetteXing1991/TA-Seq2Seq) :star::star:
* **MA**: "Mechanism-Aware Neural Machine for Dialogue Response Generation". AAAI(2017) [[PDF]](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14471/14267) :star::star:
* **HRED**: "Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models". AAAI(2016) [[PDF]](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/11957/12160) [[code]](https://github.com/julianser/hed-dlg) :star::star:
* **VHRED**: "A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues". AAAI(2017) [[PDF]](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14567/14219) [[code]](https://github.com/julianser/hed-dlg-truncated) :star::star:
* **CVAE/KgCVAE**: "Learning Discourse-level Diversity for Neural Dialog Models using Conditional Variational Autoencoders". ACL(2017) [[PDF]](https://aclweb.org/anthology/P17-1061) [[code]](https://github.com/snakeztc/NeuralDialog-CVAE) :star::star::star:
* **ERM**: "Elastic Responding Machine for Dialog Generation with Dynamically Mechanism Selecting". AAAI(2018) [[PDF]](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16316/16134) :star::star:
* **Tri-LSTM**: "Augmenting End-to-End Dialogue Systems With Commonsense Knowledge". AAAI(2018) [[PDF]](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16573/16030) :star::star:
* **CCM**: "Commonsense Knowledge Aware Conversation Generation with Graph Attention". IJCAI(2018) [[PDF]](https://www.ijcai.org/proceedings/2018/0643.pdf) [[code]](https://github.com/tuxchow/ccm) :star::star::star::star::star:
* **Retrieval+multi-seq2seq**: "An Ensemble of Retrieval-Based and Generation-Based Human-Computer Conversation Systems". IJCAI(2018) [[PDF]](https://www.ijcai.org/proceedings/2018/0609.pdf) :star::star::star:

### Personalized Dialogue
* **PAML**: "Personalizing Dialogue Agents via Meta-Learning". ACL(2019) [[PDF]](https://www.aclweb.org/anthology/P19-1542) [[code]](https://github.com/HLTCHKUST/PAML) :star::star::star:
* **PCCM**: "Assigning Personality/Profile to a Chatting Machine for Coherent Conversation Generation". IJCAI(2018) [[PDF]](https://www.ijcai.org/proceedings/2018/0595.pdf) [[code]](https://github.com/qianqiao/AssignPersonality) :star::star::star::star:
* **ECM**: "Emotional Chatting Machine: Emotional Conversation Generation with Internal and External Memory". AAAI(2018) [[PDF]](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16455/15753) [[code]](https://github.com/tuxchow/ecm) :star::star::star::star:

### Miscellaneous
* **CrossWOZ**: "CrossWOZ: A Large-Scale Chinese Cross-Domain Task-Oriented Dialogue Dataset". TACL(2020) [[PDF]](https://arxiv.org/pdf/2002.11893.pdf) [[code]](https://github.com/thu-coai/CrossWOZ) :star::star::star:
* **MultiWOZ**: "MultiWOZ - A Large-Scale Multi-Domain Wizard-of-Oz Dataset for Task-Oriented Dialogue Modelling". EMNLP(2018) [[PDF]](https://www.aclweb.org/anthology/D18-1547) [[code]](https://github.com/budzianowski/multiwoz) :star::star:
* **Survey of Dialogue**: "A Survey on Dialogue Systems: Recent Advances and New Frontiers". SIGKDD Explorations(2017) [[PDF]](https://arxiv.org/pdf/1711.01731.pdf) :star:
* **Survey of Dialogue Corpora**: "A Survey of Available Corpora For Building Data-Driven Dialogue Systems: The Journal Version". Dialogue & Discourse(2018) [[PDF]](http://dad.uni-bielefeld.de/index.php/dad/article/view/3690/3616) :star:
* **Table-to-Text Generation (R,C,T)**: "Table-to-Text Generation with Effective Hierarchical Encoder on Three Dimensions (Row, Column and Time)". EMNLP(2019) [[PDF]](https://www.aclweb.org/anthology/D19-1310.pdf) [[code]](https://github.com/ErnestGong/data2text-three-dimensions) :star::star::star:
* **LU-DST**: "Multi-task Learning for Joint Language Understanding and Dialogue State Tracking". SIGDIAL(2018) [[PDF]](https://www.aclweb.org/anthology/W18-5045)  :star::star:
* **MTask-M**: "Multi-Task Learning for Speaker-Role Adaptation in Neural Conversation Models". IJCNLP(2018) [[PDF]](https://www.aclweb.org/anthology/I17-1061) :star:
* **ADVMT**: "One “Ruler” for All Languages: Multi-Lingual Dialogue Evaluation with Adversarial Multi-Task Learning". IJCAI(2018) [[PDF]](https://www.ijcai.org/proceedings/2018/0616.pdf) :star:


## Knowledge Representation and Reasoning
* **GNTP**: "Differentiable Reasoning on Large Knowledge Bases and Natural Language". AAAI(2020) [[PDF]](https://arxiv.org/pdf/1912.10824.pdf) [[code]](https://github.com/uclnlp/gntp) :star::star::star::star:
* **NTP**: "End-to-End Differentiable Proving". NeurIPS(2017) [[PDF]](http://papers.nips.cc/paper/6969-end-to-end-differentiable-proving.pdf) [[code]](https://github.com/uclnlp/ntp) :star::star::star::star:


## Text Summarization
* **BERTSum**: "Fine-tune BERT for Extractive Summarization". arXiv(2019) [[PDF]](https://arxiv.org/pdf/1903.10318.pdf) [[code]](https://github.com/nlpyang/BertSum) :star::star::star:
* **BERT-Two-Stage**: "Pretraining-Based Natural Language Generation for Text Summarization". arXiv(2019)  [[PDF]](https://arxiv.org/pdf/1902.09243.pdf) :star::star:
* **QASumm**: "Guiding Extractive Summarization with Question-Answering Rewards". NAACL(2019) [[PDF]](https://www.aclweb.org/anthology/N19-1264) [[code]](https://github.com/ucfnlp/summ_qa_rewards) :star::star::star::star:
* **Re^3Sum**: "Retrieve, Rerank and Rewrite: Soft Template Based Neural Summarization". ACL(2018) [[PDF]](https://www.aclweb.org/anthology/P18-1015) [[code]](http://www4.comp.polyu.edu.hk/~cszqcao/data/IRSum_Resource.zip) :star::star::star:
* **NeuSum**: "Neural Document Summarization by Jointly Learning to Score and Select Sentences". ACL(2018) [[PDF]](https://www.aclweb.org/anthology/P18-1061) :star::star::star:
* **rnn-ext+abs+RL+rerank**: "Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting". ACL(2018) [[PDF]](https://www.aclweb.org/anthology/P18-1063) [[Notes]](https://www.aclweb.org/anthology/attachments/P18-1063.Notes.pdf) [[code]](https://github.com/ChenRocks/fast_abs_rl) :star::star::star::star::star:
* **Seq2Seq+CGU**: "Global Encoding for Abstractive Summarization". ACL(2018) [[PDF]](https://www.aclweb.org/anthology/P18-2027) [[code]](https://github.com/lancopku/Global-Encoding) :star::star::star:
* **ML+RL**: "A Deep Reinforced Model for Abstractive Summarization". ICLR(2018) [[PDF]](https://arxiv.org/pdf/1705.04304.pdf) :star::star::star:
* **T-ConvS2S**: "Don’t Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization". EMNLP(2018) [[PDF]](https://www.aclweb.org/anthology/D18-1206) [[code]](https://github.com/shashiongithub/XSum) :star::star::star::star:
* **RL-Topic-ConvS2S**: "A reinforced topic-aware convolutional sequence-to-sequence model for abstractive text summarization". IJCAI (2018) [[PDF]](https://www.ijcai.org/proceedings/2018/0619.pdf) :star::star::star:
* **GANsum**: "Generative Adversarial Network for Abstractive Text Summarization". AAAI(2018) [[PDF]](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16238/16492) :star:
* **FTSum**: "Faithful to the Original: Fact Aware Neural Abstractive Summarization". AAAI(2018) [[PDF]](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16121/16007) :star::star:
* **PGN**: "Get To The Point: Summarization with Pointer-Generator Networks". ACL(2017) [[PDF]](https://aclweb.org/anthology/P17-1099) [[code]](https://github.com/abisee/pointer-generator) :star::star::star::star::star:
* **ABS/ABS+**: "A Neural Attention Model for Abstractive Sentence Summarization". EMNLP(2015) [[PDF]](https://www.aclweb.org/anthology/D15-1044) :star::star:
* **RAS-Elman/RAS-LSTM**: "Abstractive Sentence Summarization with Attentive Recurrent Neural Networks". NAACL(2016) [[PDF]](https://www.aclweb.org/anthology/N16-1012) [[code]](https://github.com/facebookarchive/NAMAS)  :star::star::star:
* **words-lvt2k-1sent**: "Abstractive Text Summarization using Sequence-to-sequence RNNs and Beyond". CoNLL(2016) [[PDF]](https://www.aclweb.org/anthology/K16-1028) :star:


## Topic Modeling
* **LDA**: "Latent Dirichlet Allocation". JMLR(2003) [[PDF]](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) [[code]](https://github.com/blei-lab/lda-c) :star::star::star::star::star:
* **Parameter Estimation**: "Parameter estimation for text analysis". Technical report (2005). [[PDF]](http://www.arbylon.net/publications/text-est2.pdf) :star::star::star:
* **DTM**: "Dynamic Topic Models". ICML(2006) [[PDF]](https://dl.acm.org/citation.cfm?id=1143859) [[code]](https://github.com/blei-lab/dtm) :star::star::star::star:
* **cDTM**: "Continuous Time Dynamic Topic Models". UAI(2008) [[PDF]](https://dslpitt.org/uai/papers/08/p579-wang.pdf) :star::star:
* **iDocNADE**: "Document Informed Neural Autoregressive Topic Models with Distributional Prior". AAAI(2019) [[PDF]](https://aaai.org/ojs/index.php/AAAI/article/view/4616) [[code]](https://github.com/pgcool/iDocNADEe) :star::star::star::star:
* **NTM**: "A Novel Neural Topic Model and Its Supervised Extension". AAAI(2015) [[PDF]](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9303/9544) :star::star::star::star:
* **TWE**: "Topical Word Embeddings". AAAI(2015) [[PDF]](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9314/9535) :star::star:
* **RATM-D**: "Recurrent Attentional Topic Model". AAAI(2017)[[PDF]](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14400/14203) :star::star::star::star:
* **RIBS-TM**: "Don't Forget the Quantifiable Relationship between Words: Using Recurrent Neural Network for Short Text Topic Discovery". AAAI(2017) [[PDF]](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14172/13900) :star::star::star:
* **Topic coherence**: "Optimizing Semantic Coherence in Topic Models". EMNLP(2011) [[PDF]](https://www.aclweb.org/anthology/D11-1024) :star::star:
* **Topic coherence**: "Automatic Evaluation of Topic Coherence". NAACL(2010) [[PDF]](https://www.aclweb.org/anthology/N10-1012) :star::star:
* **DADT**: "Authorship Attribution with Author-aware Topic Models". ACL(2012) [[PDF]](https://www.aclweb.org/anthology/P12-2052) :star::star::star::star:
* **Gaussian-LDA**: "Gaussian LDA for Topic Models with Word Embeddings". ACL(2015) [[PDF]](https://www.aclweb.org/anthology/P15-1077) [[code]](https://github.com/rajarshd/Gaussian_LDA) :star::star::star::star:
* **LFTM**:	"Improving Topic Models with Latent Feature Word Representations". TACL(2015) [[PDF]](https://transacl.org/ojs/index.php/tacl/article/view/582/158) [[code]](https://github.com/datquocnguyen/LFTM) :star::star::star::star::star:
* **TopicVec**: "Generative Topic Embedding: a Continuous Representation of Documents". ACL (2016) [[PDF]](https://www.aclweb.org/anthology/P16-1063) [[code]](https://github.com/askerlee/topicvec) :star::star::star::star:
* **SLRTM**: "Sentence Level Recurrent Topic Model: Letting Topics Speak for Themselves". arXiv(2016) [[PDF]](https://arxiv.org/pdf/1604.02038.pdf) :star::star:
* **TopicRNN**: "TopicRNN: A Recurrent Neural Network with Long-Range Semantic Dependency". ICLR(2017) [[PDF]](https://arxiv.org/pdf/1611.01702.pdf) [[code]](https://github.com/dangitstam/topic-rnn) :star::star::star::star::star:
* **NMF boosted**: "Stability of topic modeling via matrix factorization". Expert Syst. Appl. (2018) [[PDF]](https://www.sciencedirect.com/science/article/pii/S0957417417305948?via%3Dihub) :star::star:
* **Evaluation of Topic Models**: "External Evaluation of Topic Models". Australasian Doc. Comp. Symp. (2009) [[PDF]](http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=471A5EE9D06BABFA4DC5CFD1E7F88A20?doi=10.1.1.529.7854&rep=rep1&type=pdf) :star::star:
* **Topic2Vec**: "Topic2Vec: Learning distributed representations of topics". IALP(2015) [[PDF]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7451564) :star::star::star:
* **L-EnsNMF**: "L-EnsNMF: Boosted Local Topic Discovery via Ensemble of Nonnegative Matrix Factorization". ICDM(2016) [[PDF]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7837872) [[code]](https://github.com/sanghosuh/lens_nmf-matlab) :star::star::star::star::star:
* **DC-NMF**: "DC-NMF: nonnegative matrix factorization based on divide-and-conquer for fast clustering and topic modeling". J. Global Optimization (2017) [[PDF]](https://link.springer.com/content/pdf/10.1007%2Fs10898-017-0515-z.pdf) :star::star::star:
* **cFTM**: "The contextual focused topic model". KDD(2012) [[PDF]](https://dl.acm.org/citation.cfm?doid=2339530.2339549) :star::star::star:
* **CLM**: "Collaboratively Improving Topic Discovery and Word Embeddings by Coordinating Global and Local Contexts". KDD(2017) [[PDF]](https://dl.acm.org/citation.cfm?doid=3097983.3098009) [[code]](https://github.com/XunGuangxu/2in1) :star::star::star::star::star:
* **GMTM**: "Unsupervised Topic Modeling for Short Texts Using Distributed Representations of Words". NAACL(2015) [[PDF]](https://www.aclweb.org/anthology/W15-1526) :star::star::star::star:
* **GPU-PDMM**: "Enhancing Topic Modeling for Short Texts with Auxiliary Word Embeddings". TOIS (2017) [[PDF]](https://dl.acm.org/citation.cfm?doid=3133943.3091108) :star::star::star:
* **BPT**: "A Two-Level Topic Model Towards Knowledge Discovery from Citation Networks". TKDE (2014) [[PDF]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6494572) :star::star::star:
* **BTM**: "A Biterm Topic Model for Short Texts". WWW(2013) [[PDF]](https://dl.acm.org/citation.cfm?doid=2488388.2488514) [[code]](https://github.com/xiaohuiyan/BTM) :star::star::star::star:
* **HGTM**: "Using Hashtag Graph-Based Topic Model to Connect Semantically-Related Words Without Co-Occurrence in Microblogs". TKDE(2016) [[PDF]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7412726) :star::star::star:
* **COTM**: "A topic model for co-occurring normal documents and short texts". WWW (2018) [[PDF]](https://link.springer.com/content/pdf/10.1007%2Fs11280-017-0467-8.pdf) :star::star::star::star:


## Machine Translation
* **Multi-pass decoder**: "Adaptive Multi-pass Decoder for Neural Machine Translation". EMNLP(2018) [[PDF]](https://www.aclweb.org/anthology/D18-1048) :star::star::star:
* **Deliberation Networks**: "Deliberation Networks: Sequence Generation Beyond One-Pass Decoding". NeurIPS(2017) [[PDF]](http://papers.nips.cc/paper/6775-deliberation-networks-sequence-generation-beyond-one-pass-decoding.pdf) :star::star::star:
* **KVMem-Attention**: "Neural Machine Translation with Key-Value Memory-Augmented Attention". IJCAI(2018) [[PDF]](https://www.ijcai.org/proceedings/2018/0357.pdf) :star::star::star::star:
* **Interactive-Attention**: "Interactive Attention for Neural Machine Translation". COLING(2016) [[PDF]](https://www.aclweb.org/anthology/C16-1205) :star::star::star:


## Question Answering
* **CFC**: "Coarse-grain Fine-grain Coattention Network for Multi-evidence Question Answering". ICLR(2019) [[PDF]](https://arxiv.org/pdf/1901.00603.pdf) :star::star:
* **MTQA**: "Multi-Task Learning with Multi-View Attention for Answer Selection and Knowledge Base Question Answering". AAAI(2019) [[PDF]](https://aaai.org/ojs/index.php/AAAI/article/view/4593) [[code]](https://github.com/dengyang17/MTQA) :star::star::star:
* **CQG-KBQA**: "Knowledge Base Question Answering via Encoding of
Complex Query Graphs". EMNLP(2018) [[PDF]](https://www.aclweb.org/anthology/D18-1242) [[code]](http://202.120.38.146/CompQA/) :star::star::star::star::star:
* **HR-BiLSTM**: "Improved Neural Relation Detection for Knowledge Base Question Answering". ACL(2017) [[PDF]](https://aclweb.org/anthology/P17-1053) :star::star::star:
* **KBQA-CGK**: "An End-to-End Model for Question Answering over Knowledge Base with Cross-Attention Combining Global Knowledge". ACL(2017) [[PDF]](https://aclweb.org/anthology/P17-1021) :star::star::star:
* **KVMem**: "Key-Value Memory Networks for Directly Reading Documents". EMNLP(2016) [[PDF]](https://www.aclweb.org/anthology/D16-1147) :star::star::star:


## Reading Comprehension
* **DecompRC**: "Multi-hop Reading Comprehension through Question Decomposition and Rescoring". ACL(2019) [[PDF]](https://www.aclweb.org/anthology/P19-1613) [[code]](https://github.com/shmsw25/DecompRC) :star::star::star::star:
* **FlowQA**: "FlowQA: Grasping Flow in History for Conversational Machine Comprehension". ICLR(2019) [[PDF]](https://arxiv.org/pdf/1810.06683.pdf) [[code]](https://github.com/momohuang/FlowQA) :star::star::star::star::star:
*  **SDNet**: "SDNet: Contextualized Attention-based Deep Network for Conversational Question Answering". arXiv(2018) [[PDF]](https://arxiv.org/pdf/1812.03593.pdf) [[code]](https://github.com/microsoft/SDNet) :star::star::star::star:


## Image Captioning
* **MLAIC**: "A Multi-task Learning Approach for Image Captioning". IJCAI(2018) [[PDF]](https://www.ijcai.org/proceedings/2018/0168.pdf) [[code]](https://github.com/andyweizhao/Multitask_Image_Captioning) :star::star::star:
* **Up-Down Attention**: "Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering". CVPR(2018) [[PDF]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Anderson_Bottom-Up_and_Top-Down_CVPR_2018_paper.pdf) :star::star::star::star:
* **SCST**: "Self-critical Sequence Training for Image Captioning". CVPR(2017) [[PDF]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8099614) :star::star::star::star:
* **Recurrent-RSA**: "Pragmatically Informative Image Captioning with Character-Level Inference". NAACL(2018) [[PDF]](https://www.aclweb.org/anthology/N18-2070) [[code]](https://github.com/reubenharry/Recurrent-RSA) :star::star::star: