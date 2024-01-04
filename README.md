# Paper-Reading
Paper reading list in 💬 **dialogue systems** and 📝 **natural language generation**. This repository will keep updating ... 🤗

- [Deep Learning in NLP](#deep-learning-in-nlp)
- [Pre-trained (Large) Language Models](#pre-trained-large-language-models)
- [Dialogue Systems](#dialogue-systems)
  - [Survey on Dialogue](#survey-on-dialogue)
  - [LLMs for Conversation](#llms-for-conversation)
  - [Multimodal Dialogue](#multimodal-dialogue)
    - [Situated and Embodied Dialogue](#situated-and-embodied-dialogue)
    - [Visually-grounded Dialogue](#visually-grounded-dialogue)
  - [Proactive Dialogue](#proactive-dialogue)
    - [Target-oriented Dialogue](#target-oriented-dialogue)
    - [Non-collaborative Dialogue (Persuasion and Negotiation)](#non-collaborative-dialogue-persuasion-and-negotiation)
  - [Personalized Dialogue](#personalized-dialogue)
    - [Character-based Dialogue](#character-based-dialogue)
    - [Personality-aware Dialogue](#personality-aware-dialogue)
    - [Persona-based Dialogue](#persona-based-dialogue)
  - [Emotional Dialogue](#emotional-dialogue)
    - [Emotional Support Dialogue](#emotional-support-dialogue)
    - [Empathetic Dialogue](#empathetic-dialogue)
  - [Recommendation Dialogue and CRS](#recommendation-dialogue-and-crs)
  - [Knowledge-grounded Dialogue](#knowledge-grounded-dialogue)
  - [Task-oriented Dialogue](#task-oriented-dialogue)
  - [Open-domain Dialogue](#open-domain-dialogue)
  - [Dialogue Evaluation](#dialogue-evaluation)
- [Natural Language Generation](#natural-language-generation)
  - [Survey on NLG](#survey-on-nlg)
  - [NLG Theories and Techniques](#nlg-theories-and-techniques)
  - [Diffusion Models for NLG](#diffusion-models-for-nlg)
  - [Controllable Generation](#controllable-generation)
  - [Text Planning](#text-planning)
  - [Decoding Algorithms](#decoding-algorithms)

***

## Deep Learning in NLP
* **iNLP**: "Interactive Natural Language Processing". arXiv(2023) [[paper]](https://arxiv.org/abs/2305.13246)  :star::star::star::star:
* **Data Augmentation**: "A Survey of Data Augmentation Approaches for NLP". ACL-Findings(2021) [[paper]](https://arxiv.org/abs/2105.03075)
* **Prompting**: "Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing". arXiv(2021) [[paper]](https://arxiv.org/abs/2107.13586)  :star::star::star::star::star:
* **NLP World Scope**: "Experience Grounds Language". EMNLP(2020) [[paper]](https://aclanthology.org/2020.emnlp-main.703/)  :star::star::star::star::star:
* **Graphormer**: "Do Transformers Really Perform Bad for Graph Representation?". NeurIPS(2021) [[paper]](https://arxiv.org/abs/2106.05234) [[code]](https://github.com/Microsoft/Graphormer)
* **GAT**: "Graph Attention Networks". ICLR(2018) [[paper]](https://arxiv.org/paper/1710.10903.paper) [[code-tf]](https://github.com/PetarV-/GAT) [[code-py]](https://github.com/Diego999/pyGAT)
* **Transformer-XL**: "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context". ACL(2019) [[paper]](https://www.aclweb.org/anthology/P19-1285) [[code]](https://github.com/kimiyoung/transformer-xl)
* **Transformer**: "Attention is All you Need". NeurIPS(2017) [[paper]](https://papers.nips.cc/paper/7181-attention-is-all-you-need.paper) [[code-official]](https://github.com/tensorflow/tensor2tensor) [[code-tf]](https://github.com/Kyubyong/transformer) [[code-py]](https://github.com/jadore801120/attention-is-all-you-need-pytorch) :star::star::star::star::star:
* **VAE**: "An Introduction to Variational Autoencoders". arXiv(2019) [[paper]](https://arxiv.org/paper/1906.02691.paper)
* **Survey on Attention**: "An Introductory Survey on Attention Mechanisms in NLP Problems". arXiv(2018) [[paper]](https://arxiv.org/paper/1811.05544.paper) :star::star::star::star::star:
* **Additive Attention**: "Neural Machine Translation by Jointly Learning to Align and Translate". ICLR(2015) [[paper]](https://arxiv.org/paper/1409.0473.paper) 
* **Multiplicative Attention**: "Effective Approaches to Attention-based Neural Machine Translation". EMNLP(2015) [[paper]](https://www.aclweb.org/anthology/D15-1166)
* **Memory Net**: "End-To-End Memory Networks". NeurIPS(2015) [[paper]](https://papers.nips.cc/paper/5846-end-to-end-memory-networks.paper)
* **Copy Mechanism (PGN)**: "Get To The Point: Summarization with Pointer-Generator Networks". ACL(2017) [[paper]](https://aclweb.org/anthology/P17-1099) [[code]](https://github.com/abisee/pointer-generator) :star::star::star::star::star:
* **Copy Mechanism**: "Incorporating Copying Mechanism in Sequence-to-Sequence Learning". ACL(2016) [[paper]](https://www.aclweb.org/anthology/P16-1154)
* **ELMo**: "Deep contextualized word representations". NAACL(2018) [[paper]](https://www.aclweb.org/anthology/N18-1202) [[code]](https://github.com/allenai/bilm-tf)
* **Glove**: "GloVe: Global Vectors for Word Representation". EMNLP(2014) [[paper]](https://www.aclweb.org/anthology/D14-1162.paper) [[code]](https://github.com/stanfordnlp/GloVe)
* **word2vec**: "word2vec Parameter Learning Explained". arXiv(2016) [[paper]](https://arxiv.org/paper/1411.2738.paper) :star::star::star::star::star:
* **Multi-task Learning**: "An Overview of Multi-Task Learning in Deep Neural Networks". arXiv(2017) [[paper]](https://arxiv.org/paper/1706.05098.paper)
* **Gradient Descent**: "An Overview of Gradient Descent Optimization Algorithms". arXiv(2016) [[paper]](https://arxiv.org/paper/1609.04747.paper) :star::star::star::star::star:

👆 [Back to Top](#paper-reading)


## Pre-trained (Large) Language Models

* **Llama 2**: "Llama 2: Open Foundation and Fine-Tuned Chat Models". Meta(2023) [[paper]](https://arxiv.org/abs/2307.09288) [[model]](https://ai.meta.com/resources/models-and-libraries/llama/) [[code]](https://github.com/facebookresearch/llama)
* **GPT-4**: "GPT-4 Technical Report". OpenAI(2023) [[paper]](https://arxiv.org/abs/2303.08774) :star::star::star::star:
* **LLaMA**: "LLaMA: Open and Efficient Foundation Language Models". Meta(2023) [[paper]](https://arxiv.org/abs/2302.13971) [[code]](https://github.com/facebookresearch/llama/tree/llama_v1) :star::star::star::star:
* **OPT**: "OPT: Open Pre-trained Transformer Language Models". arXiv(2022) [[paper]](https://arxiv.org/abs/2205.01068) [[code]](https://github.com/facebookresearch/metaseq)
* **PaLM**: "PaLM: Scaling Language Modeling with Pathways". arXiv(2022) [[paper]](https://arxiv.org/abs/2204.02311)
* **Survey of PLMs**: "Pre-Trained Models: Past, Present and Future". arXiv(2021) [[paper]](https://arxiv.org/abs/2106.07139)
* **Survey of PLMs**: "Pre-trained Models for Natural Language Processing: A Survey". arXiv(2020) [[paper]](https://arxiv.org/paper/2003.08271.paper)
* **GLM-130B**: "GLM-130B: An Open Bilingual Pre-trained Model". ICLR(2023) [[paper]](https://arxiv.org/abs/2210.02414) [[code]](https://github.com/THUDM/GLM-130B) [[Blog]](http://keg.cs.tsinghua.edu.cn/glm-130b/zh/posts/glm-130b/) :star::star::star:
* **GLM**: "GLM: General Language Model Pretraining with Autoregressive Blank Infilling". ACL(2022) [[paper]](https://arxiv.org/abs/2103.10360) [[code]](https://github.com/THUDM/GLM)
* **GPT-3**: "Language Models are Few-Shot Learners". NeurIPS(2020) [[paper]](https://arxiv.org/abs/2005.14165) :star::star::star::star:
* **BART**: "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension". ACL(2020) [[paper]](https://www.aclweb.org/anthology/2020.acl-main.703.paper) [[code]](https://github.com/huggingface/transformers) :star::star::star:
* **T5**: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer". JMLR(2020) [[paper]](https://arxiv.org/abs/1910.10683) [[code-tf]](https://github.com/google-research/text-to-text-transfer-transformer) [[code-py]](https://github.com/huggingface/transformers) :star::star::star:
* **ALBERT**: "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations". ICLR(2020) [[paper]](https://openreview.net/paper?id=H1eA7AEtvS)
* **TinyBERT**: "TinyBERT: Distilling BERT for Natural Language Understanding". arXiv(2019) [[paper]](https://arxiv.org/paper/1909.10351.paper) [[code]](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT)
* **Chinese BERT**: "Pre-Training with Whole Word Masking for Chinese BERT". arXiv(2019) [[paper]](https://arxiv.org/paper/1906.08101.paper) [[code]](https://github.com/ymcui/Chinese-BERT-wwm)
* **SpanBERT**: "SpanBERT: Improving Pre-training by Representing and Predicting Spans". TACL(2020) [[paper]](https://arxiv.org/paper/1907.10529.paper) [[code]](https://github.com/facebookresearch/SpanBERT)
* **RoBERTa**: "RoBERTa: A Robustly Optimized BERT Pretraining Approach". arXiv(2019) [[paper]](https://arxiv.org/paper/1907.11692.paper) [[code]](https://github.com/pytorch/fairseq)
* **UniLM**: "Unified Language Model Pre-training for
Natural Language Understanding and Generation". NeurIPS(2019) [[paper]](https://papers.nips.cc/paper/9464-unified-language-model-pre-training-for-natural-language-understanding-and-generation.paper) [[code]](https://github.com/microsoft/unilm) :star::star::star:
* **XLNet**: "XLNet: Generalized Autoregressive Pretraining for Language Understanding". NeurIPS(2019) [[paper]](https://papers.nips.cc/paper/8812-xlnet-generalized-autoregressive-pretraining-for-language-understanding.paper) [[code]](https://github.com/zihangdai/xlnet)
* **GPT-2**: "Language Models are Unsupervised Multitask Learners". OpenAI(2019) [[paper]](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.paper) [[code]](https://huggingface.co/gpt2)
* **BERT**: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding". NAACL(2019) [[paper]](https://www.aclweb.org/anthology/N19-1423) [[code]](https://github.com/google-research/bert) :star::star::star::star::star:
* **GPT**: "Improving Language Understanding by Generative Pre-Training". OpenAI(2018) [[paper]](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.paper) :star::star::star::star::star:

👆 [Back to Top](#paper-reading)


## Dialogue Systems

### Survey on Dialogue
* **Proactive Dialogue**: "A Survey on Proactive Dialogue Systems: Problems, Methods, and Prospects". IJCAI(2023) [[paper]](https://arxiv.org/abs/2305.02750)
* **Responsible Dialogue**: "Recent Advances towards Safe, Responsible, and Moral Dialogue Systems: A Survey". arXiv(2023) [[paper]](https://arxiv.org/abs/2302.09270)
* **Negotiation Dialogue**: "Let's Negotiate! A Survey of Negotiation Dialogue Systems". arXiv(2022) [[paper]](https://arxiv.org/abs/2212.09072)
* **DL-based Dialogue**: "Recent Advances in Deep Learning Based Dialogue Systems: A Systematic Survey". arXiv(2021) [[paper]](https://arxiv.org/abs/2105.04387) :star::star::star::star:
* **Open-domain Dialogue**: "Challenges in Building Intelligent Open-domain Dialog Systems". TOIS(2020) [[paper]](https://dl.acm.org/doi/10.1145/3383123)
* **Dialogue Systems**: "A Survey on Dialogue Systems: Recent Advances and New Frontiers". SIGKDD Explorations(2017) [[paper]](https://arxiv.org/paper/1711.01731.paper)
* **Dialogue Corpora**: "A Survey of Available Corpora For Building Data-Driven Dialogue Systems". arXiv(2017) [[paper]](https://arxiv.org/paper/1512.05742.paper) [[data]](https://breakend.github.io/DialogDatasets/)

👆 [Back to Top](#paper-reading)


### LLMs for Conversation
* **Parrot**: "Parrot: Enhancing Multi-Turn Chat Models by Learning to Ask Questions". arXiv(2023) [[paper]](https://arxiv.org/abs/2310.07301)
* **MemoChat**: "MemoChat: Tuning LLMs to Use Memos for Consistent Long-Range Open-Domain Conversation". arXiv(2023) [[paper]](https://arxiv.org/abs/2308.08239)
* **Llama 2-Chat**: "Llama 2: Open Foundation and Fine-Tuned Chat Models". Meta(2023) [[paper]](https://arxiv.org/abs/2307.09288) [[code]](https://github.com/facebookresearch/llama)
* **ChatGLM3**: "ChatGLM3 Series: Open Bilingual Chat LLMs". Tsinghua(2023) [[code]](https://github.com/THUDM/ChatGLM3)
* **ChatGLM2-6B**: "ChatGLM2-6B: An Open Bilingual Chat LLM". Tsinghua(2023) [[code]](https://github.com/THUDM/ChatGLM2-6B)
* **MPC**: "Prompted LLMs as Chatbot Modules for Long Open-domain Conversation". ACL-Findings(2023) [[paper]](https://aclanthology.org/2023.findings-acl.277) [[code]](https://github.com/krafton-ai/MPC)
* **MemoryBank-SiliconFriend**: "MemoryBank: Enhancing Large Language Models with Long-Term Memory". arXiv(2023) [[paper]](https://arxiv.org/abs/2305.10250) [[code]](https://github.com/zhongwanjun/MemoryBank-SiliconFriend)
* **UltraChat**: "Enhancing Chat Language Models by Scaling High-quality Instructional Conversations". arXiv(2023) [[paper]](https://arxiv.org/abs/2305.14233) [[data]](https://github.com/thunlp/UltraChat)
* **ChatAlpaca**: "ChatAlpaca: A Multi-Turn Dialogue Corpus based on Alpaca Instructions". Github(2023) [[data]](https://github.com/cascip/ChatAlpaca)
* **Phoenix**: "Phoenix: Democratizing ChatGPT across Languages". arXiv(2023) [[paper]](https://arxiv.org/abs/2304.10453) [[code]](https://github.com/FreedomIntelligence/LLMZoo)
* **Dolly**: "Free Dolly: Introducing the World's First Truly Open Instruction-Tuned LLM". Databricks(2023) [[code]](https://github.com/databrickslabs/dolly)
* **Baize**: "Baize: An Open-Source Chat Model with Parameter-Efficient Tuning on Self-Chat Data". arXiv(2023) [[paper]](https://arxiv.org/abs/2304.01196) [[code]](https://github.com/project-baize/baize-chatbot)
* **Vicuna**: "Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90% ChatGPT Quality". LMSYS Org(2023) [[Blog]](https://vicuna.lmsys.org/) [[code]](https://github.com/lm-sys/FastChat)
* **Koala**: "Koala: A Dialogue Model for Academic Research". UC Berkeley(2023) [[Blog]](https://bair.berkeley.edu/blog/2023/04/03/koala/) [[code]](https://github.com/young-geng/EasyLM)
* **BELLE**: "BELLE: Be Everyone's Large Language model Engine". LianjiaTech(2023) [[code]](https://github.com/LianjiaTech/BELLE)
* **Alpaca**: "Alpaca: A Strong, Replicable Instruction-Following Model". Stanford(2023) [[Blog]](https://crfm.stanford.edu/2023/03/13/alpaca.html) [[code]](https://github.com/tatsu-lab/stanford_alpaca) [[alpaca-lora]](https://github.com/tloen/alpaca-lora)
* **ChatGLM-6B**: "An Open Bilingual Dialogue Language Model". Tsinghua(2023) [[code]](https://github.com/THUDM/ChatGLM-6B)
* **Open-Assistant**: "Open Assistant: Conversational AI for everyone". Github(2023) [[project]](https://open-assistant.io/) [[code]](https://github.com/LAION-AI/Open-Assistant)
* **ChatGPT**: "ChatGPT: Optimizing Language Models for Dialogue". OpenAI(2022) [[Blog]](https://openai.com/blog/chatgpt/) :star::star::star::star::star:
* **Sparrow**: "Improving alignment of dialogue agents via targeted human judgements". arXiv(2022) [[paper]](https://arxiv.org/abs/2209.14375) [[data]](https://storage.googleapis.com/deepmind-media/DeepMind.com/Authors-Notes/sparrow/sparrow.html)
* **BlenderBot3**: "BlenderBot 3: a deployed conversational agent that continually learns to responsibly engage". arXiv(2022) [[paper]](https://arxiv.org/abs/2208.03188)
* **LaMDA**: "LaMDA: Language Models for Dialog Applications". arXiv(2022) [[paper]](https://arxiv.org/abs/2201.08239)
* **GODEL**: "GODEL: Large-Scale Pre-Training for Goal-Directed Dialog". arXiv(2022) [[paper]](https://arxiv.org/abs/2206.11309) [[code]](https://github.com/Microsoft/GODEL)
* **Anthropic Assistant-v2**: "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback". arXiv(2022) [[paper]](https://arxiv.org/abs/2204.05862)
* **Anthropic Assistant**: "A General Language Assistant as a Laboratory for Alignment". arXiv(2021) [[paper]](https://arxiv.org/abs/2112.00861)

👆 [Back to Top](#paper-reading)


### Multimodal Dialogue

#### Situated and Embodied Dialogue
* **Emb-Plan**: "Multimodal Embodied Plan Prediction Augmented with Synthetic Embodied Dialogue". EMNLP(2023) [[paper]](https://aclanthology.org/2023.emnlp-main.374)
* **WTaG**: "Can Foundation Models Watch, Talk and Guide You Step by Step to Make a Cake?". EMNLP-Findings(2023) [[paper]](https://arxiv.org/abs/2311.00738) [[code]]()
* **SIMMC-VR**: "SIMMC-VR: A Task-oriented Multimodal Dialog Dataset with Situated and Immersive VR Streams". ACL(2023) [[paper]](https://aclanthology.org/2023.acl-long.345/) :star::star::star::star:
* **SURE**: "Multimodal Recommendation Dialog with Subjective Preference: A New Challenge and Benchmark". ACL(2023) [[paper]](https://arxiv.org/abs/2305.18212) [[data]](https://github.com/LYX0501/SURE)
* **SUGAR**: "A Textual Dataset for Situated Proactive Response Selection". ACL(2023) [[paper]](https://aclanthology.org/2023.acl-long.214/) [[data]](https://github.com/notani/sugar-conversational-dataset)
* **Chat-with-NeRF**: "Chat with NeRF: Grounding 3D Objects in Neural Radiance Field through Dialog". UMich(2023) [[project]](https://chat-with-nerf.github.io/) [[code]](https://github.com/sled-group/chat-with-nerf)
* **MindDial**: "MindDial: Belief Dynamics Tracking with Theory-of-Mind Modeling for Situated Neural Dialogue Generation". arXiv(2023) [[paper]](https://arxiv.org/abs/2306.15253)
* **Collab**: "Towards Collaborative Plan Acquisition through Theory of Mind Modeling in Situated Dialogue". IJCAI(2023) [[paper]](https://arxiv.org/abs/2305.11271) [[code]](https://github.com/sled-group/collab-plan-acquisition)
* **SEAGULL**: "SEAGULL: An Embodied Agent for Instruction Following through Situated Dialog". Alexa Prize SimBot Challenge(2023) [[paper]](https://assets.amazon.science/17/c5/a684745e4d6d94742d85a31e5362/simbot-challenge-technical-report-seagull-1.paper)
* **SitCoM-DETR**: "Which One Are You Referring To? Multimodal Object Identification in Situated Dialogue". EACL-SRW(2023) [[paper]](https://arxiv.org/abs/2302.14680) [[code]](https://github.com/holylovenia/multimodal-object-identification)
* **MLR**: "Improving Situated Conversational Agents with Step-by-Step Multi-modal Logic Reasoning". DSTC11(2023) [[paper]](https://aclanthology.org/2023.dstc-1.3/)
* **SimpleMTOD**: "SimpleMTOD: A Simple Language Model for Multimodal Task-Oriented Dialogue with Symbolic Scene Representation". arXiv(2023) [[paper]](https://arxiv.org/abs/2307.04907)
* **SPRING**: "SPRING: Situated Conversation Agent Pretrained with Multimodal Questions from Incremental Layout Graph". AAAI(2023) [[paper]](https://arxiv.org/abs/2301.01949) [[code]](https://github.com/LYX0501/SPRING)
* **DOROTHIE**: "DOROTHIE: Spoken Dialogue for Handling Unexpected Situations in Interactive Autonomous Driving Agents". EMNLP-Findings(2022) [[paper]](http://arxiv.org/abs/2210.12511) [[code]](https://github.com/sled-group/DOROTHIE) :star::star::star::star:
* **LIGHT-curriculum**: "Situated Dialogue Learning through Procedural Environment Generation". ACL(2022) [[paper]](https://arxiv.org/abs/2110.03262)
* **DANLI**: "DANLI: Deliberative Agent for Following Natural Language Instructions". EMNLP(2022) [[paper]](https://arxiv.org/abs/2210.12485) [[code]](https://github.com/sled-group/danli)
* **PRS**: "Learning to Mediate Disparities Towards Pragmatic Communication". ACL(2022) [[paper]](http://arxiv.org/abs/2203.13685) [[code]](https://github.com/sled-group/Pragmatic-Rational-Speaker)
* **Joint-model**: "Learning to Embed Multi-Modal Contexts for Situated Conversational Agents". NAACL-Findings(2022) [[paper]](https://aclanthology.org/2022.findings-naacl.61) [[code]](https://github.com/KAIST-AILab/DSTC10-SIMMC)
* **TEACh_FILM**: "Don't Copy the Teacher: Data and Model Challenges in Embodied Dialogue". EMNLP(2022) [[paper]](http://arxiv.org/abs/2210.04443) [[code]](https://github.com/soyeonm/TEACh_FILM)
* **TEACh**: "TEACh: Task-driven Embodied Agents that Chat". AAAI(2022) [[paper]](http://arxiv.org/abs/2110.00534) [[data]](https://github.com/alexa/teach) 
* **MindCraft**: "MindCraft: Theory of Mind Modeling for Situated Dialogue in Collaborative Tasks". EMNLP(2021) [[paper]](https://aclanthology.org/2021.emnlp-main.85/) [[code]](https://github.com/sled-group/MindCraft) :star::star::star::star:
* **Multimodal-model**: "Multimodal Interactions Using Pretrained Unimodal Models for SIMMC 2.0". DSTC10(2022) [[paper]](https://arxiv.org/abs/2112.05328) [[code]](https://github.com/rungjoo/simmc2.0)
* **SIMMC 2.0**: "SIMMC 2.0: A Task-oriented Dialog Dataset for Immersive Multimodal Conversations" EMNLP(2021) [[paper]](https://aclanthology.org/2021.emnlp-main.401/) [[code]](https://github.com/facebookresearch/simmc2)  :star::star::star::star:
* **MM-DST**: "Multi-Task Learning for Situated Multi-Domain End-to-End Dialogue Systems". arXiv(2021) [[paper]](https://arxiv.org/abs/2110.05221)
* **SIMMC**: "Situated and Interactive Multimodal Conversations". COLING(2020) [[paper]](https://arxiv.org/abs/2006.01460) [[code]](https://github.com/facebookresearch/simmc)
* **Minecraft-BAP**: "Learning to execute instructions in a Minecraft dialogue". ACL(2020) [[paper]](https://aclanthology.org/2020.acl-main.232) [[code]](https://github.com/prashant-jayan21/minecraft-bap-models)
* **Minecraft Dialogue**: "Collaborative Dialogue in Minecraft". ACL(2019) [[paper]](https://aclanthology.org/P19-1537) [[code]](https://github.com/prashant-jayan21/minecraft-dialogue-models)
* **CLG**: "Collaborative Language Grounding Toward Situated Human‐Robot Dialogue". AI Magazine(2016) [[paper]](https://ojs.aaai.org/aimagazine/index.php/aimagazine/article/view/2684) :star::star::star::star:
* **SHRD**: "Back to the Blocks World: Learning New Actions through Situated Human-Robot Dialogue". SIGDIAL(2014) [[paper]](https://aclanthology.org/W14-4313/)


#### Visually-grounded Dialogue
* **VLAW-MDM**: "A Framework for Vision-Language Warm-up Tasks in Multimodal Dialogue Models". EMNLP(2023) [[paper]](https://aclanthology.org/2023.emnlp-main.167/) [[code]](https://github.com/BeneciaLee/VLAW-MDM)
* **VDialogUE**: "VDialogUE: A Unified Evaluation Benchmark for Visually-grounded Dialogue". arXiv(2023) [[paper]](https://arxiv.org/abs/2309.07387)
* **TextBind**: "TextBind: Multi-turn Interleaved Multimodal Instruction-following in the Wild". arXiv(2023) [[paper]](https://arxiv.org/abs/2309.08637) [[data]](https://github.com/SihengLi99/TextBind)
* **VSTAR**: "VSTAR: A Video-grounded Dialogue Dataset for Situated Semantic Understanding with Scene and Topic Transitions". ACL(2023) [[paper]](https://arxiv.org/abs/2305.18756) [[data]](https://vstar-benchmark.github.io/)
* **ComSet**: "Multimodal Persona Based Generation of Comic Dialogs". ACL(2023) [[paper]](https://aclanthology.org/2023.acl-long.791/) [[code]](https://github.com/dair-iitd/MPdialog)
* **MPCHAT**: "MPCHAT: Towards Multimodal Persona-Grounded Conversation". ACL(2023) [[paper]](https://arxiv.org/abs/2305.17388) [[code]](https://github.com/ahnjaewoo/mpchat)
* **PaCE**: "PaCE: Unified Multi-modal Dialogue Pre-training with Progressive and Compositional Experts". ACL(2023) [[paper]](https://arxiv.org/abs/2305.14839) [[code]](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/pace)
* **MMDialog**: "MMDialog: A Large-scale Multi-turn Dialogue Dataset Towards Multi-modal Open-domain Conversation". ACL(2023) [[paper]](https://arxiv.org/abs/2211.05719) [[data]](https://github.com/victorsungo/MMDialog) :star::star::star:
* **MDS-S2**: "Dual Semantic Knowledge Composed Multimodal Dialog Systems". SIGIR(2023) [[paper]](https://arxiv.org/abs/2305.09990)
* **TikTalk**: "TikTalk: A Multi-Modal Dialogue Dataset for Real-World Chitchat". arXiv(2023) [[paper]](https://arxiv.org/abs/2301.05880) [[code]](https://github.com/RUC-AIMind/TikTalk)
* **CHAMPAGNE**: "CHAMPAGNE: Learning Real-world Conversation from Large-Scale Web Videos". arXiv(2023) [[paper]](https://arxiv.org/abs/2303.09713) [[code]](https://seungjuhan.me/champagne/)
* **MMChat**: "MMChat: Multi-Modal Chat Dataset on Social Media". LREC(2022) [[paper]](https://arxiv.org/abs/2108.07154) [[code]](https://github.com/silverriver/MMChat)
* **CRVD**: "Collaborative Reasoning on Multi-Modal Semantic Graphs for Video-Grounded Dialogue Generation". EMNLP-Findings(2022) [[paper]](https://arxiv.org/abs/2210.12460)
* **M3ED**: "M3ED: Multi-modal Multi-scene Multi-label Emotional Dialogue Database". ACL(2022) [[paper]](https://aclanthology.org/2022.acl-long.391) [[data]](https://github.com/AIM3-RUC/RUCM3ED)
* **MDRG**: "Multimodal Dialogue Response Generation". ACL(2022) [[paper]](https://arxiv.org/abs/2110.08515)
* **UniTranSeR**: "UniTranSeR: A Unified Transformer Semantic Representation Framework for Multimodal Task-Oriented Dialog System". ACL(2022) [[paper]](https://aclanthology.org/2022.acl-long.9)
* **PhotoChat**: "PhotoChat: A Human-Human Dialogue Dataset With Photo Sharing Behavior For Joint Image-Text Modeling". ACL(2021) [[paper]](https://aclanthology.org/2021.acl-long.479/) [[data]](https://github.com/google-research/google-research/tree/master/multimodalchat)
* **Multi-Modal Dialogue**: "Constructing Multi-Modal Dialogue Dataset by Replacing Text with Semantically Relevant Images". ACL(2021) [[paper]](http://arxiv.org/abs/2107.08685) [[code]](https://github.com/shh1574/multi-modal-dialogue-dataset)
* **OpenViDial 2.0**: "OpenViDial 2.0: A Larger-Scale, Open-Domain Dialogue Generation Dataset with Visual Contexts". arXiv(2021) [[paper]](https://arxiv.org/abs/2109.12761) [[data]](https://github.com/ShannonAI/OpenViDial)
* **MMConv**: "MMConv: An Environment for Multimodal Conversational Search across Multiple Domains". SIGIR(2021) [[paper]](https://dl.acm.org/doi/10.1145/3404835.3462970) [[data]](https://github.com/liziliao/MMConv)
* **Image Chat**: "Image Chat: Engaging Grounded Conversations". ACL(2020) [[paper]](https://arxiv.org/abs/1811.00945) [[data]](https://parl.ai/projects/image_chat/)
* **MTN**: "Multimodal Transformer Networks for End-to-End Video-Grounded Dialogue Systems". ACL(2019) [[paper]](https://arxiv.org/abs/1907.01166) [[code]](https://github.com/henryhungle/MTN) :star::star::star:
* **MELD**: "MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations". ACL(2019) [[paper]](https://aclanthology.org/P19-1050) [[data]](https://affective-meld.github.io/)
* **CLEVR-Dialog**: "CLEVR-Dialog: A Diagnostic Dataset for Multi-Round Reasoning in Visual Dialog". NAACL(2019) [[paper]](https://arxiv.org/abs/1903.03166) [[data]](https://github.com/satwikkottur/clevr-dialog)
* **VisDial-RL**: "Improving Generative Visual Dialog by Answering Diverse Questions". EMNLP(2019) [[paper]](https://arxiv.org/abs/1909.10470) [[code]](https://github.com/vmurahari3/visdial-diversity)
* **MMD**: "Towards Building Large Scale Multimodal Domain-Aware Conversation Systems". AAAI(2018) [[paper]](https://arxiv.org/abs/1704.00200) [[code]](https://amritasaha1812.github.io/MMD/)
* **Talk the Walk**: "Talk the Walk: Navigating New York City through Grounded Dialogue". arXiv(2018) [[paper]](https://arxiv.org/abs/1807.03367) [[code]](https://github.com/facebookresearch/talkthewalk)
* **VisDial**: "Visual Dialog". CVPR(2017) [[paper]](https://arxiv.org/abs/1611.08669) [[code]](https://github.com/batra-mlp-lab/visdial)


👆 [Back to Top](#paper-reading)


### Proactive Dialogue

#### Target-oriented Dialogue
* **TopDial**: "Target-oriented Proactive Dialogue Systems with Personalization: Problem Formulation and Dataset Curation". EMNLP(2023) [[paper]](https://arxiv.org/abs/2310.07397) [[code]](https://github.com/iwangjian/TopDial)
* **RTCP**: "Reinforced Target-driven Conversational Promotion". EMNLP(2023) [[paper]](https://aclanthology.org/2023.emnlp-main.775/) [[code]](https://github.com/huyquangdao/RTCP)
* **ProCoT**: "Prompting and Evaluating Large Language Models for Proactive Dialogues: Clarification, Target-guided, and Non-collaboration". EMNLP-Findings(2023) [[paper]](https://arxiv.org/abs/2305.13626) [[code]](https://github.com/dengyang17/LLM-Proactive)
* **MTGP**: "MTGP: Multi-turn Target-oriented Dialogue Guided by Generative Global Path with Flexible Turns". ACL-Findings(2023) [[paper]](https://aclanthology.org/2023.findings-acl.18/) [[code]](https://github.com/sxnohnarla/MTGP)
* **COLOR**: "Dialogue Planning via Brownian Bridge Stochastic Process for Goal-directed Proactive Dialogue". ACL-Findings(2023) [[paper]](https://arxiv.org/abs/2305.05290) [[code]](https://github.com/iwangjian/Color4Dial)  :star::star::star:
* **TopKG**: "TopKG: Target-oriented Dialog via Global Planning on Knowledge Graph". COLING(2022) [[paper]](https://aclanthology.org/2022.coling-1.62) [[code]](https://github.com/yyyyyyzt/topkgchat)
* **TGCP**: "Target-Guided Open-Domain Conversation Planning". COLING(2022) [[paper]](https://arxiv.org/abs/2209.09746) [[code]](https://github.com/y-kishinami/TGCP)
* **FOP**: "Long-term Control for Dialogue Generation: Methods and Evaluation". NAACL(2022) [[paper]](https://arxiv.org/abs/2205.07352) [[code]](https://github.com/asappresearch/constrained-dialogue-generation)
* **CODA**: "Target-Guided Dialogue Response Generation Using Commonsense and Data Augmentation". NAACL-Findings(2022) [[paper]](https://arxiv.org/abs/2205.09314) [[code]](https://github.com/prakharguptaz/target-guided-dialogue-coda)
* **OTTers**: "OTTers: One-turn Topic Transitions for Open-Domain Dialogue". ACL(2021) [[paper]](https://arxiv.org/abs/2105.13710) [[data]](https://github.com/karinseve/OTTers)
* **CG-nAR**: "Thinking Clearly, Talking Fast: Concept-Guided Non-Autoregressive Generation for Open-Domain Dialogue Systems". EMNLP(2021) [[paper]](https://arxiv.org/abs/2109.04084) [[code]](https://github.com/RowitZou/CG-nAR) :star::star::star:
* **DuConv**: "Proactive Human-Machine Conversation with Explicit Conversation Goals". ACL(2019) [[paper]](https://www.aclweb.org/anthology/P19-1369) [[code]](https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2019-DuConv)
* **CKC**: "Keyword-Guided Neural Conversational Model". AAAI(2021) [[paper]](https://arxiv.org/abs/2012.08383) [[code]](https://github.com/zhongpeixiang/CKC)
* **KnowHRL**: "Knowledge Graph Grounded Goal Planning for Open-Domain Conversation Generation". AAAI(2020) [[paper]](https://aaai.org/ojs/index.php/AAAI/article/view/6474)
* **DKRN**: "Dynamic Knowledge Routing Network For Target-Guided Open-Domain Conversation". AAAI(2020) [[paper]](https://arxiv.org/abs/2002.01196) [[code]](https://github.com/James-Yip/TGODC-DKRN)
* **TGConv**: "Target-Guided Open-Domain Conversation". ACL(2019) [[paper]](https://aclanthology.org/P19-1565/) [[code]](https://github.com/squareRoot3/Target-Guided-Conversation)


#### Non-collaborative Dialogue (Persuasion and Negotiation)
* **INA**: "INA: An Integrative Approach for Enhancing Negotiation Strategies with Reward-Based Dialogue System". EMNLP(2023) [[paper]](https://arxiv.org/abs/2310.18207) [[data]](https://github.com/zishan-ai/neg)
* **I-Pro**: "Interacting with Non-Cooperative User: A New Paradigm for Proactive Dialogue Policy". SIGIR(2022) [[paper]](https://arxiv.org/abs/2204.07433)
* **PAAD**: "Towards a Progression-Aware Autonomous Dialogue Agent". NAACL(2022) [[paper]](https://arxiv.org/abs/2205.03692) [[code]](https://github.rpi.edu/LACAI/dialogue-progression) :star::star::star:
* **PersRFI**: "Refine and Imitate: Reducing Repetition and Inconsistency in Persuasion Dialogues via Reinforcement Learning and Human Demonstration". EMNLP-Findings(2021) [[paper]](https://arxiv.org/abs/2012.15375) [[code]](https://github.com/wyshi/consistency)
* **ResPer**: "RESPER: Computationally Modelling Resisting Strategies in Persuasive Conversations". EACL(2021) [[paper]](https://arxiv.org/abs/2101.10545) [[code]](https://github.com/americast/resper)
* **ARDM**: "Alternating Recurrent Dialog Model with Large-scale Pre-trained Language Models". EACL(2021) [[paper]](https://aclanthology.org/2021.eacl-main.110) [[code]](https://github.com/qywu/ARDM)
* **DialoGraph**: "DialoGraph: Incorporating Interpretable Strategy-Graph Networks into Negotiation Dialogues". ICLR(2021) [[paper]](https://arxiv.org/abs/2106.00920) [[code]](https://github.com/rishabhjoshi/DialoGraph_ICLR21) :star::star::star:
* **NegotiationToM**: "Improving Dialog Systems for Negotiation with Personality Modeling". ACL(2021) [[paper]](https://arxiv.org/abs/2010.09954) [[code]](https://github.com/princeton-nlp/NegotiationToM)
* **FeHED**: "Augmenting Non-Collaborative Dialog Systems with Explicit Semantic and Strategic Dialog History". ICLR(2020) [[paper]](https://openreview.net/forum?id=ryxQuANKPB) [[code]](https://github.com/zhouyiheng11/augmenting-non-collabrative-dialog)
* **CTX-PSA**: "Learning to Plan and Realize Separately for Open-Ended Dialogue Systems". EMNLP-Findings(2020) [[paper]](https://arxiv.org/abs/2009.12506) [[code]](https://github.com/sashank06/planning_generation)
* **Negotiation-Coach**: "A Dynamic Strategy Coach for Effective Negotiation". SIGDIAL(2019) [[paper]](https://aclanthology.org/W19-5943/) [[code]](https://github.com/zhouyiheng11/Negotiation-Coach)
* **PersuasionForGood**: "Persuasion for Good: Towards a Personalized Persuasive Dialogue System for Social Good". ACL(2019) [[paper]](https://aclanthology.org/P19-1566) [[data]](https://gitlab.com/ucdavisnlp/persuasionforgood)
* **CraigslistBargain**: "Decoupling Strategy and Generation in Negotiation Dialogues". EMNLP(2018) [[paper]](https://aclanthology.org/D18-1256/) [[data]](https://stanfordnlp.github.io/cocoa/)

👆 [Back to Top](#paper-reading)


### Personalized Dialogue

#### Character-based Dialogue
* **LLM-Werewolf**: "Exploring Large Language Models for Communication Games: An Empirical Study on Werewolf". arXiv(2023) [[paper]](https://arxiv.org/abs/2309.04658)
* **ChatHaruhi**: "ChatHaruhi: Reviving Anime Character in Reality via Large Language Model". arXiv(2023) [[report]](https://arxiv.org/abs/2308.09597) [[code]](https://github.com/LC1332/Chat-Haruhi-Suzumiya)
* **DPCD**: "Hi Sheldon! Creating Deep Personalized Characters from TV Shows". arXiv(2023) [[paper]](https://arxiv.org/abs/2304.11093) [[data]](https://github.com/Metaverse-AI-Lab-THU/Deep-Personalized-Character-Dataset-DPCD)
* **Cornell-Rich**: "Personalised Language Modelling of Screen Characters Using Rich Metadata Annotations". arXiv(2023) [[paper]](https://arxiv.org/abs/2303.16618) [[data]](https://github.com/st-vincent1/cornell_rich)
* **KNUDGE**: "Ontologically Faithful Generation of Non-Player Character Dialogues". arXic(2022) [[paper]](https://arxiv.org/abs/2212.10618)
* **HPD**: "Large Language Models Meet Harry Potter: A Bilingual Dataset for Aligning Dialogue Agents with Characters". arXiv(2022) [[paper]](https://arxiv.org/abs/2211.06869) [[data]](https://nuochenpku.github.io/HPD.github.io/)
* **DialStory**: "A Benchmark for Understanding and Generating Dialogue between Characters in Stories". arXiv(2022) [[paper]](https://arxiv.org/abs/2209.08524)
* **CareCall**: "Building a Role Specified Open-Domain Dialogue System Leveraging Large-Scale Language Models". NAACL(2022) [[paper]](https://arxiv.org/abs/2205.00176) [[data]](https://github.com/naver-ai/carecall-corpus)
* **PDP**: "Meet Your Favorite Character: Open-domain Chatbot Mimicking Fictional Characters with only a Few Utterances". NAACL(2022) [[paper]](https://arxiv.org/abs/2204.10825) [[code]](https://github.com/hyperconnect/pseudo-dialog-prompting)
* **RPA**: "Am I Me or You? State-of-the-Art Dialogue Models Cannot Maintain an Identity". NAACL-Findings(2022) [[paper]](https://aclanthology.org/2022.findings-naacl.182/)
* **CharacterChat**: "CharacterChat: Supporting the Creation of Fictional Characters through Conversation and Progressive Manifestation with a Chatbot". ACM C&C(2021）[[paper]](https://arxiv.org/abs/2106.12314)
* **ALOHA**: "ALOHA: Artificial Learning of Human Attributes for Dialogue Agents". AAAI(2020) [[paper]](https://arxiv.org/abs/1910.08293) [[code]](https://github.com/newpro/aloha-chatbot)
* **LIGHT**: "Learning to Speak and Act in a Fantasy Text Adventure Game". EMNLP(2019) [[paper]](https://aclanthology.org/D19-1062/) [[data]](https://parl.ai/projects/light/) :star::star::star:

#### Personality-aware Dialogue
* **CharacterChat**: "CharacterChat: Learning towards Conversational AI with Personalized Social Support". arXiv(2023) [[paper]](https://arxiv.org/abs/2308.10278) [[code]](https://github.com/morecry/characterchat)
* **ChatGPT-MBTI**: "Can ChatGPT Assess Human Personalities? A General Evaluation Framework". arXiv(2023) [[paper]](https://arxiv.org/abs/2303.01248) [[code]](https://github.com/Kali-Hac/ChatGPT-MBTI)
* **Prompted Personality**: "Controlling Personality Style in Dialogue with Zero-Shot Prompt-Based Learning". IWSDS(2023) [[paper]](https://arxiv.org/abs/2302.03848)
* **CPED**: "CPED: A Large-Scale Chinese Personalized and Emotional Dialogue Dataset for Conversational AI". arXiv(2022) [[paper]](https://arxiv.org/abs/2205.14727) [[data]](https://github.com/scutcyr/CPED) :star::star::star:
* **PELD**: "Automatically Select Emotion for Response via Personality-affected Emotion Transition". ACL-Findings(2021) [[paper]](https://arxiv.org/abs/2106.15846) [[data]](https://github.com/preke/PELD)
* **FriendsPersona**: "Automatic Text-based Personality Recognition on Monologues and Multiparty Dialogues Using Attentive Networks and Contextual Embeddings". AAAI-Student Abstract(2020) [[paper]](https://arxiv.org/abs/1911.09304) [[data]](https://github.com/emorynlp/personality-detection)
* **APR**: "Identifying Personality Traits Using Overlap Dynamics in Multiparty Dialogue". INTERSPEECH(2019) [[paper]](https://arxiv.org/abs/1909.00876)
* **PersonalDilaog**: "Personalized Dialogue Generation with Diversified Traits". arXiv(2019) [[paper]](https://arxiv.org/abs/1901.09672) [[data]](https://github.com/silverriver/PersonalDilaog)
* **PersonageNLG**: "Controlling Personality-Based Stylistic Variation with Neural Natural Language Generators". SIGDIAL(2018) [[paper]](https://arxiv.org/abs/1805.08352) [[data]](https://nlds.soe.ucsc.edu/stylistic-variation-nlg)

#### Persona-based Dialogue
* **VaRMI**: "Building Persona Consistent Dialogue Agents with Offline Reinforcement Learning". EMNLP(2023) [[paper]](https://arxiv.org/abs/2310.10735) [[code]](https://github.com/ryanshea10/personachat_offline_rl)
* **OPELA**: "When Crowd Meets Persona: Creating a Large-Scale Open-Domain Persona Dialogue Corpus". arXiv(2023) [[paper]](https://arxiv.org/abs/2304.00350) [[data]](https://github.com/smilegate-ai/OPELA)
* **ORIG**: "Towards Robust Personalized Dialogue Generation via Order-Insensitive Representation Regularization". ACL-Findings(2023) [[paper]](https://arxiv.org/abs/2305.12782) [[code]](https://github.com/ChanLiang/ORIG)
* **CLV**: "Enhancing Personalized Dialogue Generation with Contrastive Latent Variables: Combining Sparse and Dense Persona". ACL(2023) [[paper]](https://arxiv.org/abs/2305.11482) [[code]](https://github.com/Toyhom/CLV)
* **SimOAP**: "SimOAP: Improve Coherence and Consistency in Persona-based Dialogue Generation via Over-sampling and Post-evaluation". ACL(2023) [[paper]](https://arxiv.org/abs/2305.11130) [[code]](https://github.com/934865517zjk/SimOAP)
* **LMEDR**: "Learning to Memorize Entailment and Discourse Relations for Persona-Consistent Dialogues". AAAI(2023) [[paper]](https://arxiv.org/abs/2301.04871) [[code]](https://github.com/Chenrj233/LMEDR)
* **Retrieval-to-Prediction**: "Improving Personality Consistency in Conversation by Persona Extending". CIKM(2022) [[paper]](https://arxiv.org/abs/2208.10816) [[code]](https://github.com/CCIIPLab/Persona_Extend)
* **Implicit-Persona**: "A Personalized Dialogue Generator with Implicit User Persona Detection". COLING(2022) [[paper]](https://arxiv.org/abs/2204.07372)
* **CareCallMemory**: "Keep Me Updated! Memory Management in Long-term Conversations". EMNLP-Findings(2022) [[paper]](https://arxiv.org/abs/2210.08750) [[data]](https://github.com/naver-ai/carecall-memory)
* **PersonaDefense**: "You Don't Know My Favorite Color: Preventing Dialogue Representations from Revealing Speakers' Private Personas". NAACL(2022) [[paper]](https://arxiv.org/abs/2205.10228) [[code]](https://github.com/HKUST-KnowComp/Persona_leakage_and_defense_in_GPT-2)
* **Prompt-Tuning**: "Building a Personalized Dialogue System with Prompt-Tuning". NAACL-SRW(2022) [[paper]](https://arxiv.org/abs/2206.05399)
* **DuLeMon**: "Long Time No See! Open-Domain Conversation with Long-Term Persona Memory". ACL-Findings(2022) [[paper]](https://arxiv.org/abs/2203.05797) [[data]](https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2022-DuLeMon) :star::star::star:
* **INFO**: "You Truly Understand What I Need: Intellectual and Friendly Dialogue Agents grounding Knowledge and Persona". EMNLP-Findings(2022) [[paper]](https://arxiv.org/abs/2301.02401) [[code]](https://github.com/dlawjddn803/INFO)
* **FoCus**: "Call for Customized Conversation: Customized Conversation Grounding Persona and Knowledge". AAAI(2022) [[paper]](https://arxiv.org/abs/2112.08619) [[code]](https://github.com/pkchat-focus/FoCus) :star::star::star:
* **MSP**: "Less is More: Learning to Refine Dialogue History for Personalized Dialogue Generation". NAACL(2022) [[paper]](https://aclanthology.org/2022.naacl-main.426/)
* **GME**: "Transferable Persona-Grounded Dialogues via Grounded Minimal Edits". EMNLP(2021) [[paper]](https://arxiv.org/abs/2109.07713) [[code]](https://github.com/thu-coai/grounded-minimal-edit)
* **BoB**: "BoB: BERT Over BERT for Training Persona-based Dialogue Models from Limited Personalized Data". ACL(2021) [[paper]](https://aclanthology.org/2021.acl-long.14) [[code]](https://github.com/songhaoyu/BoB)
* **PABST**: "Unsupervised Enrichment of Persona-grounded Dialog with Background Stories". ACL(2021) [[paper]](https://arxiv.org/abs/2106.08364) [[code]](https://github.com/majumderb/pabst)
* **DHAP**: "One Chatbot Per Person: Creating Personalized Chatbots based on Implicit User Profiles". SIGIR(2021) [[paper]](https://dl.acm.org/doi/10.1145/3404835.3462828)
* **Pchatbot**: "Pchatbot: A Large-Scale Dataset for Personalized Chatbot". SIGIR(2021) [[paper]](http://arxiv.org/abs/2009.13284) [[data]](https://github.com/qhjqhj00/SIGIR2021-Pchatbot) :star::star::star:
* **COMPAC**: "Like hiking? You probably enjoy nature: Persona-grounded Dialog with Commonsense Expansions". EMNLP(2020) [[paper]](https://arxiv.org/abs/2010.03205) [[code]](https://github.com/majumderb/compac)
* **pragmatic-consistency**: "Will I Sound Like Me? Improving Persona Consistency in Dialogues through Pragmatic Self-Consciousness". EMNLP(2020) [[paper]](https://arxiv.org/abs/2004.05816) [[code]](https://github.com/skywalker023/pragmatic-consistency) :star::star::star::star:
* **XPersona**: "XPersona: Evaluating Multilingual Personalized Chatbot". arXiv(2020) [[paper]](https://arxiv.org/abs/2003.07568) [[data]](https://github.com/HLTCHKUST/Xpersona)
* **KvPI**: "Profile Consistency Identification for Open-domain Dialogue Agents". EMNLP(2020) [[paper]](https://aclanthology.org/2020.emnlp-main.539/) [[code]](https://github.com/songhaoyu/KvPI)
* **GDR**: "Generate, Delete and Rewrite: A Three-Stage Framework for Improving Persona Consistency of Dialogue Generation". ACL(2020) [[paper]](https://aclanthology.org/2020.acl-main.516/)
* **P^2Bot**: "You Impress Me: Dialogue Generation via Mutual Persona Perception". ACL(2020) [[paper]](https://aclanthology.org/2020.acl-main.131) [[code]](https://github.com/SivilTaram/Persona-Dialogue-Generation)
* **RCDG**: "Generating Persona Consistent Dialogues by Exploiting Natural Language Inference". AAAI(2020) [[paper]](https://arxiv.org/abs/1911.05889) [[code]](https://github.com/songhaoyu/RCDG)
* **Persona-sparse**: "A Pre-training Based Personalized Dialogue Generation Model with Persona-sparse Data". AAAI(2020) [[paper]](https://arxiv.org/abs/1911.04700)
* **PersonaWAE**: "Modeling Personalization in Continuous Space for Response Generation via Augmented Wasserstein Autoencoders". EMNLP(2019) [[paper]](https://aclanthology.org/D19-1201/)
* **PAML**: "Personalizing Dialogue Agents via Meta-Learning". ACL(2019) [[paper]](https://www.aclweb.org/anthology/P19-1542) [[code]](https://github.com/HLTCHKUST/PAML)
* **PersonaChat**: "Personalizing Dialogue Agents: I have a dog, do you have pets too?" ACL(2018) [[paper]](https://aclanthology.org/P18-1205) [[data]](https://github.com/facebookresearch/ParlAI/tree/main/projects/personachat) :star::star::star:
* **PCCM**: "Assigning Personality/Profile to a Chatting Machine for Coherent Conversation Generation". IJCAI(2018) [[paper]](https://www.ijcai.org/proceedings/2018/0595.paper)

👆 [Back to Top](#paper-reading)


### Emotional Dialogue

#### Emotional Support Dialogue
* **KEMI**: "Knowledge-enhanced Mixed-initiative Dialogue System for Emotional Support Conversations". ACL(2023) [[paper]](https://arxiv.org/abs/2305.10172) [[code]](https://github.com/dengyang17/KEMI) :star::star::star::star:
* **CSConv**: "A Cognitive Stimulation Dialogue System with Multi-source Knowledge Fusion for Elders with Cognitive Impairment". ACL(2023) [[paper]](https://arxiv.org/abs/2305.08200) [[code]](https://github.com/jiangjyjy/CSD)
* **AugESC**: "AugESC: Dialogue Augmentation with Large Language Models for Emotional Support Conversation". ACL-Findings(2023) [[paper]](https://arxiv.org/abs/2202.13047)
* **TransESC**: "TransESC: Smoothing Emotional Support Conversation via Turn-Level State Transition". ACL-Findings(2023) [[paper]](https://arxiv.org/abs/2305.03296) [[code]](https://github.com/circle-hit/TransESC)
* **PAL**: "PAL: Persona-Augmented Emotional Support Conversation Generation". arXiv(2022) [[paper]](https://arxiv.org/abs/2212.09235)
* **MultiESC**: "Improving Multi-turn Emotional Support Dialogue Generation with Lookahead Strategy Planning". EMNLP(2022) [[paper]](https://arxiv.org/abs/2210.04242) [[code]](https://github.com/lwgkzl/MultiESC) :star::star::star::star:
* **MISC**: "MISC: A MIxed Strategy-Aware Model Integrating COMET for Emotional Support Conversation". ACL(2022) [[paper]](https://arxiv.org/abs/2203.13560) [[code]](https://github.com/morecry/MISC)
* **C3KG**: "C3KG: A Chinese Commonsense Conversation Knowledge Graph". ACL-Findings(2022) [[paper]](https://arxiv.org/abs/2204.02549) [[data]](https://github.com/XiaoMi/C3KG)
* **GLHG**: "Control Globally, Understand Locally: A Global-to-Local Hierarchical Graph Network for Emotional Support Conversation". IJCAI(2022) [[paper]](https://arxiv.org/abs/2204.12749)
* **ESConv**: "Towards Emotional Support Dialog Systems". ACL(2021) [[paper]](https://arxiv.org/abs/2106.01144) [[data]](https://github.com/thu-coai/Emotional-Support-Conversation) :star::star::star::star:


#### Empathetic Dialogue
* **E-CORE**: "E-CORE: Emotion Correlation Enhanced Empathetic Dialogue Generation" EMNLP(2023) [[paper]](https://aclanthology.org/2023.emnlp-main.653)
* **EmpSOA**: "Don't Lose Yourself! Empathetic Response Generation via Explicit Self-Other Awareness". ACL-Findings(2023) [[paper]](https://arxiv.org/abs/2210.03884) [[code]](https://github.com/circle-hit/EmpSOA)
* **CASE**: "CASE: Aligning Coarse-to-Fine Cognition and Affection for Empathetic Response Generation". ACL(2023) [[paper]](https://arxiv.org/abs/2208.08845) [[code]](https://github.com/jfzhouyoo/CASE)
* **CARE**: "CARE: Causality Reasoning for Empathetic Responses by Conditional Graph Generation". EMNLP-Findings(2022) [[paper]](https://arxiv.org/abs/2211.00255) [[code]](https://github.com/wangjs9/CARE-master)
* **PosEmoDial**: "Towards Multi-Turn Empathetic Dialogs with Positive Emotion Elicitation". arXiV(2022) [[paper]](https://arxiv.org/abs/2204.10509)
* **CEM**: "CEM: Commonsense-aware Empathetic Response Generation". AAAI(2022) [[paper]](https://arxiv.org/abs/2109.05739) [[code]](https://github.com/Sahandfer/CEM)
* **GEE**: "Perspective-taking and Pragmatics for Generating Empathetic Responses Focused on Emotion Causes". EMNLP(2021) [[paper]](https://arxiv.org/abs/2109.08828) [[code]](https://github.com/skywalker023/focused-empathy)
* **RecEC**: "Improving Empathetic Response Generation by Recognizing Emotion Cause in Conversations". EMNLP-Findings(2021) [[paper]](https://aclanthology.org/2021.findings-emnlp.70) [[code]](https://github.com/A-Rain/EmpDialogue_RecEC)
* **CoMAE**: "CoMAE: A Multi-factor Hierarchical Framework for Empathetic Response Generation". ACL-Findings(2021) [[paper]](https://aclanthology.org/2021.findings-acl.72) [[code]](https://github.com/chujiezheng/CoMAE)
* **CARE**: "CARE: Commonsense-Aware Emotional Response Generation with Latent Concepts". AAAI(2021) [[paper]](https://arxiv.org/abs/2012.08377) [[code]](https://github.com/zhongpeixiang/CARE)
* **EmpDG**: "EmpDG: Multi-resolution Interactive Empathetic Dialogue Generation". COLING(2020) [[paper]](https://aclanthology.org/2020.coling-main.394) [[code]](https://github.com/qtli/EmpDG)
* **MIME**: "MIME: MIMicking Emotions for Empathetic Response Generation". EMNLP(2020) [[paper]](https://arxiv.org/abs/2010.01454) [[code]](https://github.com/declare-lab/MIME)
* **PEC**: "Towards Persona-Based Empathetic Conversational Models". EMNLP(2020) [[paper]](https://aclanthology.org/2020.emnlp-main.531) [[code]](https://github.com/zhongpeixiang/PEC)
* **MoEL**: "MoEL: Mixture of Empathetic Listeners". EMNLP(2019) [[paper]](https://aclanthology.org/D19-1012) [[code]](https://github.com/HLTCHKUST/MoEL)
* **EmpatheticDialogues**: "Towards Empathetic Open-domain Conversation Models: A New Benchmark and Dataset". ACL(2019) [[paper]](https://aclanthology.org/P19-1534) [[data]](https://github.com/facebookresearch/EmpatheticDialogues) :star::star::star:
* **EmoDS**: "Generating Responses with a Specific Emotion in Dialog". ACL(2019) [[paper]](https://aclanthology.org/P19-1359)
* **MojiTalk**: "MojiTalk: Generating Emotional Responses at Scale". ACL(2018) [[paper]](https://aclanthology.org/P18-1104)
* **ECM**: "Emotional Chatting Machine: Emotional Conversation Generation with Internal and External Memory". AAAI(2018) [[paper]](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16455/15753) [[code]](https://github.com/tuxchow/ecm)

👆 [Back to Top](#paper-reading)


### Recommendation Dialogue and CRS
* **TCP-Dial**: "Follow Me: Conversation Planning for Target-driven Recommendation Dialogue Systems". arXiv(2022) [[paper]](https://arxiv.org/abs/2208.03516) [[code]](https://github.com/iwangjian/Plan4RecDial)
* **KERS**: "KERS: A Knowledge-Enhanced Framework for Recommendation Dialog Systems with Multiple Subgoals". EMNLP-Findings(2021) [[paper]](https://aclanthology.org/2021.findings-emnlp.94) [[code]](https://github.com/z562/KERS)
* **DuRecDial2.0**: "DuRecDial 2.0: A Bilingual Parallel Corpus for Conversational Recommendation". EMNLP(2021) [[paper]](https://arxiv.org/abs/2109.08877) [[code]](https://github.com/liuzeming01/DuRecDial)
* **DuRecDial**: "Towards Conversational Recommendation over Multi-Type Dialogs". ACL(2020) [[paper]](https://arxiv.org/paper/2005.03954.paper) [[code]](https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2020-DuRecDial) :star::star::star::star:
* **TG-ReDial**: "Towards Topic-Guided Conversational Recommender System". COLING(2020) [[paper]](https://www.aclweb.org/anthology/2020.coling-main.365.paper) [[code]](https://github.com/RUCAIBox/TG-ReDial)
* **INSPIRED**: "INSPIRED: Toward Sociable Recommendation Dialog Systems". EMNLP(2020) [[paper]](https://www.aclweb.org/anthology/2020.emnlp-main.654.paper) [[data]](https://github.com/sweetpeach/Inspired)
* **GoRecDial**: "Recommendation as a Communication Game: Self-Supervised Bot-Play for Goal-oriented Dialogue". EMNLP(2019) [[paper]](https://www.aclweb.org/anthology/D19-1203.paper) [[code]](https://github.com/facebookresearch/ParlAI)
* **CRS-Survey**: "A Survey on Conversational Recommender Systems". ACM Computing Surveys(2021) [[paper]](https://arxiv.org/abs/2004.00646)
* **CRS-Survey**: "Advances and Challenges in Conversational Recommender Systems: A Survey
". arXiv(2021) [[paper]](https://arxiv.org/abs/2101.09459)
* **CRSLab**: "CRSLab: An Open-Source Toolkit for Building Conversational Recommender System". arXiv(2021) [[paper]](https://arxiv.org/paper/2101.00939.paper) [[code]](https://github.com/RUCAIBox/CRSLab) :star::star::star:
* **MESE**: "Improving Conversational Recommendation Systems' Quality with Context-Aware Item Meta Information". NAACL(2022) [[paper]](https://arxiv.org/abs/2112.08140) [[code]](https://github.com/by2299/MESE)
* **C2-CRS**: "C2-CRS: Coarse-to-Fine Contrastive Learning for Conversational Recommender System". WSDM(2022) [[paper]](https://arxiv.org/abs/2201.02732) [[code]](https://github.com/RUCAIBox/WSDM2022-C2CRS)
* **BotPlay**: "Self-Supervised Bot Play for Conversational Recommendation with Justifications". arXiv(2021) [[paper]](https://arxiv.org/abs/2112.05197)
* **RID**: "Finetuning Large-Scale Pre-trained Language Models for Conversational Recommendation with Knowledge Graph". arXiv(2021) [[paper]](https://arxiv.org/abs/2110.07477) [[code]](https://github.com/Lingzhi-WANG/PLM-BasedCRS)
* **CRFR**: "CRFR: Improving Conversational Recommender Systems via Flexible Fragments Reasoning on Knowledge Graphs". EMNLP(2021) [[paper]](https://aclanthology.org/2021.emnlp-main.355)
* **NTRD**: "Learning Neural Templates for Recommender Dialogue System". EMNLP(2021) [[paper]](https://arxiv.org/abs/2109.12302) [[code]](https://github.com/jokieleung/NTRD)
* **CR-Walker**: "CR-Walker: Tree-Structured Graph Reasoning and Dialog Acts for Conversational Recommendation". EMNLP(2021) [[paper]](https://arxiv.org/abs/2010.10333) [[code]](https://github.com/truthless11/CR-Walker) :star::star::star::star:
* **RevCore**: "RevCore: Review-augmented Conversational Recommendation". ACL-Findings(2021) [[paper]](https://arxiv.org/abs/2106.00957) [[code]](https://github.com/JD-AI-Research-NLP/RevCore)
* **KECRS**: "KECRS: Towards Knowledge-Enriched Conversational Recommendation System". arXiv(2021) [[paper]](https://arxiv.org/abs/2105.08261)
* **FPAN**: "Adapting User Preference to Online Feedback in Multi-round Conversational Recommendation". WSDM(2021) [[paper]](https://dl.acm.org/doi/10.1145/3437963.3441791) [[code]](https://github.com/xxkkrr/FPAN)
* **UNICORN**: "Unified Conversational Recommendation Policy Learning via Graph-based Reinforcement Learning". SIGIR(2021) [[paper]](https://arxiv.org/abs/2105.09710) [[code]](https://github.com/dengyang17/unicorn)
* **KGSF**: "Improving Conversational Recommender Systems via Knowledge Graph based Semantic Fusion". KDD(2020) [[paper]](https://arxiv.org/paper/2007.04032.paper) [[code]](https://github.com/RUCAIBox/KGSF)
* **CPR**: "Interactive Path Reasoning on Graph for Conversational Recommendation". KDD(2020) [[paper]](https://arxiv.org/abs/2007.00194) [[code]](https://cpr-conv-rec.github.io/)
* **EAR**: "Estimation-Action-Reflection: Towards Deep Interaction Between Conversational and Recommender Systems". WSDM(2020) [[paper]](https://arxiv.org/abs/2002.09102) [[code]](https://ear-conv-rec.github.io/)
* **KBRD**: "Towards Knowledge-Based Recommender Dialog System". EMNLP(2019) [[paper]](https://www.aclweb.org/anthology/D19-1189.paper) [[code]](https://github.com/THUDM/KBRD)
* **ReDial**: "Towards Deep Conversational Recommendations". NeurIPS(2018) [[paper]](https://papers.nips.cc/paper/8180-towards-deep-conversational-recommendations.paper) [[data]](https://github.com/ReDialData/website)

👆 [Back to Top](#paper-reading)


### Knowledge-grounded Dialogue
* **DOCTOR**: "Dialogue Chain-of-Thought Distillation for Commonsense-aware Conversational Agents". EMNLP(2023) [[paper]](https://arxiv.org/abs/2310.09343) [[code]](https://github.com/kyle8581/DialogueCoT) [[demo]](https://dialoguecot.web.app/) :star::star::star::star:
* **GATE**: "Well Begun is Half Done: Generator-agnostic Knowledge Pre-Selection for Knowledge-Grounded Dialogue". EMNLP(2023) [[paper]](https://arxiv.org/abs/2310.07659) [[code]](https://github.com/qinlang14/GATE)
* **CONNER**: "Beyond Factuality: A Comprehensive Evaluation of Large Language Models as Knowledge Generators". EMNLP(2023) [[paper]](https://arxiv.org/abs/2310.07289) [[code]](https://github.com/ChanLiang/CONNER)
* **K-DIAL**: "Improving Factual Consistency for Knowledge-Grounded Dialogue Systems via Knowledge Enhancement and Alignment". EMNLP-Findings(2023) [[paper]](https://arxiv.org/abs/2310.08372)
* **GLM-Dialog**: "GLM-Dialog: Noise-tolerant Pre-training for Knowledge-grounded Dialogue Generation". arXiv(2023) [[paper]](https://arxiv.org/abs/2302.14401) [[code]](https://github.com/RUCKBReasoning/GLM-Dialog)
* **RHO**: "RHO (ρ): Reducing Hallucination in Open-domain Dialogues with Knowledge Grounding". ACL-Findings(2023) [[paper]](https://aclanthology.org/2023.findings-acl.275.paper) [[code]](https://github.com/ziweiji/RHO) 
* **MultiRefKGC**: "There Is No Standard Answer: Knowledge-Grounded Dialogue Generation with Adversarial Activated Multi-Reference Learning". EMNLP(2022) [[paper]](https://arxiv.org/abs/2210.12459) [[code]](https://github.com/TingchenFu/MultiRefKGC) :star::star::star:
* **CorefDiffs**: "CorefDiffs: Co-referential and Differential Knowledge Flow in Document Grounded Conversations". COLING(2022) [[paper]](https://arxiv.org/abs/2210.02223) [[code]](https://github.com/cathyxl/coref-diffs)
* **DTR**: "Stylized Knowledge-Grounded Dialogue Generation via Disentangled Template Rewriting". NAACL(2022) [[paper]](https://arxiv.org/abs/2204.05610) [[code]](https://github.com/victorsungo/SKDG-DTR)
* **XDAI**: "XDAI: A Tuning-free Framework for Exploiting Pre-trained Language Models in Knowledge Grounded Dialogue Generation". KDD(2022) [[paper]](https://dl.acm.org/doi/10.1145/3534678.3539135) [[code]](https://github.com/THUDM/XDAI)
* **PersonaKGC**: "There Are a Thousand Hamlets in a Thousand People's Eyes: Enhancing Knowledge-grounded Dialogue with Personal Memory". ACL(2022) [[paper]](https://arxiv.org/abs/2204.02624) [[code]](https://github.com/Lucasftc/PersonaKGC)
* **KI**: "Lexical Knowledge Internalization for Neural Dialog Generation". ACL(2022) [[paper]](https://arxiv.org/abs/2205.01941) [[code]](https://github.com/lividwo/ki)
* **DiffKG**: "Towards Large-Scale Interpretable Knowledge Graph Reasoning for Dialogue Systems". ACL-Findings(2022) [[paper]](https://arxiv.org/abs/2203.10610) [[code]](https://github.com/Pascalson/DiffKG-Dialog) :star::star::star:
* **KSAM**: "KSAM: Infusing Multi-Source Knowledge into Dialogue Generation via Knowledge Source Aware Multi-Head Decoding". ACL-Findings(2022) [[paper]](https://aclanthology.org/2022.findings-acl.30)
* **MDSP**: "Multi-Stage Prompting for Knowledgeable Dialogue Generation". ACL-Findings(2022) [[paper]](https://arxiv.org/abs/2203.08745) [[code]](https://github.com/NVIDIA/Megatron-LM)
* **FSB**: "Few-Shot Bot: Prompt-Based Learning for Dialogue Systems". arXiv(2021) [[paper]](https://arxiv.org/abs/2110.08118) [[code]](https://github.com/andreamad8/FSB) :star::star::star:
* **P-GDG**: "Exploring Prompt-based Few-shot Learning for Grounded Dialog Generation". arXiv(2021) [[paper]](https://arxiv.org/abs/2109.06513)
* **KAT-TSLF**: "A Three-Stage Learning Framework for Low-Resource Knowledge-Grounded Dialogue Generation". EMNLP(2021) [[paper]](https://arxiv.org/abs/2109.04096) [[code]](https://github.com/neukg/KAT-TSLF)
* **DIALKI**: "DIALKI: Knowledge Identification in Conversational Systems through Dialogue-Document Contextualization". EMNLP(2021) [[paper]](https://aclanthology.org/2021.emnlp-main.140) [[code]](https://github.com/ellenmellon/DIALKI)
* **CoLV**: "CoLV: A Collaborative Latent Variable Model for Knowledge-Grounded Dialogue Generation". EMNLP(2021) [[paper]](https://aclanthology.org/2021.emnlp-main.172)
* **SKT-KG**: "Augmenting Knowledge-grounded Conversations with Sequential Knowledge Transition". NAACL(2021) [[paper]](https://www.aclweb.org/anthology/2021.naacl-main.446)
* **MSKE**: "More is Better: Enhancing Open-Domain Dialogue Generation via Multi-Source Heterogeneous Knowledge". EMNLP(2021) [[paper]](https://aclanthology.org/2021.emnlp-main.175) [[code]](https://github.com/pku-sixing/EMNLP2021-MSKE_Dialog)
* **EARL**: "EARL: Informative Knowledge-Grounded Conversation Generation with Entity-Agnostic Representation Learning". EMNLP(2021) [[paper]](https://aclanthology.org/2021.emnlp-main.184) [[code]](https://github.com/thu-coai/earl)
* **KGD-CF**: "Increasing Faithfulness in Knowledge-Grounded Dialogue with Controllable Features". ACL(2021) [[paper]](https://aclanthology.org/2021.acl-long.58.paper)
* **SECE**: "Space Efficient Context Encoding for Non-Task-Oriented Dialogue Generation with Graph Attention Transformer". ACL(2021) [[paper]](https://aclanthology.org/2021.acl-long.546) [[code]](https://github.com/fabiangal/space-efficient-context-encoding-acl21) :star::star::star:
* **MIKe**: "Initiative-Aware Self-Supervised Learning for Knowledge-Grounded Conversations". SIGIR(2021) [[paper]](https://dl.acm.org/doi/10.1145/3404835.3462824) [[code]](https://github.com/ChuanMeng/MIKe)
* **GOKC**: "Learning to Copy Coherent Knowledge for Response Generation". AAAI(2021) [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/17486) [[code]](https://github.com/jq2276/Learning2Copy)
* **KnowledGPT**: "Knowledge-Grounded Dialogue Generation with Pre-trained Language Models". EMNLP(2020) [[paper]](https://www.aclweb.org/anthology/2020.emnlp-main.272) [[code]](https://github.com/zhaoxlpku/KnowledGPT)
* **DiffKS**: "Difference-aware Knowledge Selection for Knowledge-grounded Conversation Generation". EMNLP-Findings(2020) [[paper]](https://www.aclweb.org/anthology/2020.findings-emnlp.11) [[code]](https://github.com/chujiezheng/DiffKS)
* **DukeNet**: "DukeNet: A Dual Knowledge Interaction Network for Knowledge-Grounded Conversation". SIGIR(2020) [[paper]](https://dl.acm.org/doi/10.1145/3397271.3401097) [[code]](https://github.com/ChuanMeng/DukeNet)
* **CCN**: "Cross Copy Network for Dialogue Generation". EMNLP(2020) [[paper]](https://www.aclweb.org/anthology/2020.emnlp-main.149) [[code]](https://github.com/jichangzhen/CCN)
* **PIPM**: "Bridging the Gap between Prior and Posterior Knowledge Selection for Knowledge-Grounded Dialogue Generation". EMNLP(2020) [[paper]](https://www.aclweb.org/anthology/2020.emnlp-main.275)
* **ConceptFlow**: "Grounded Conversation Generation as Guided Traverses in Commonsense Knowledge Graphs". ACL(2020) [[paper]](https://aclanthology.org/2020.acl-main.184/) [[code]](https://github.com/thunlp/ConceptFlow) :star::star::star::star:
* **ConKADI**: "Diverse and Informative Dialogue Generation with Context-Specific Commonsense Knowledge Awareness". ACL(2020) [[paper]](https://www.aclweb.org/anthology/2020.acl-main.515) [[code]](https://github.com/pku-sixing/ACL2020-ConKADI) :star::star::star:
* **KIC**: "Generating Informative Conversational Response using Recurrent Knowledge-Interaction and Knowledge-Copy". ACL(2020) [[paper]](https://www.aclweb.org/anthology/2020.acl-main.6)
* **SKT**: "Sequential Latent Knowledge Selection for Knowledge-Grounded Dialogue". ICLR(2020) [[paper]](https://openreview.net/paper?id=Hke0K1HKwr) [[code]](https://github.com/bckim92/sequential-knowledge-transformer) :star::star::star:
* **KdConv**: "KdConv: A Chinese Multi-domain Dialogue Dataset Towards Multi-turn Knowledge-driven Conversation". ACL(2020) [[paper]](https://arxiv.org/paper/2004.04100.paper) [[data]](https://github.com/thu-coai/KdConv)
* **TransDG**: "Improving Knowledge-aware Dialogue Generation via Knowledge Base Question Answering". AAAI(2020) [[paper]](https://arxiv.org/abs/1912.07491) [[code]](https://github.com/siat-nlp/TransDG)
* **RefNet**: "RefNet: A Reference-aware Network for Background Based Conversation". AAAI(2020) [[paper]](https://arxiv.org/paper/1908.06449.paper) [[code]](https://github.com/ChuanMeng/RefNet)
* **GLKS**: "Thinking Globally, Acting Locally: Distantly Supervised Global-to-Local Knowledge Selection for Background Based Conversation". AAAI(2020) [[paper]](https://arxiv.org/paper/1908.09528.paper) [[code]](https://github.com/PengjieRen/GLKS)
* **AKGCM**: "Knowledge Aware Conversation Generation with Explainable Reasoning over Augmented Graphs". EMNLP(2019) [[paper]](https://aclanthology.org/D19-1187.paper) [[code]](https://github.com/PaddlePaddle/Research/tree/master/NLP/EMNLP2019-AKGCM)
* **DyKgChat**: "DyKgChat: Benchmarking Dialogue Generation Grounding on Dynamic Knowledge Graphs". EMNLP(2019) [[paper]](https://aclanthology.org/D19-1194.paper) [[code]](https://github.com/Pascalson/DyKGChat)
* **OpenDialKG**: "OpenDialKG: Explainable Conversational Reasoning with Attention-based Walks over Knowledge Graphs". ACL(2019) [[paper]](https://www.aclweb.org/anthology/P19-1081) [[data]](https://github.com/facebookresearch/opendialkg)
* **WoW**: "Wizard of Wikipedia: Knowledge-Powered Conversational agents". ICLR(2019) [[paper]](https://arxiv.org/paper/1811.01241.paper)
* **PostKS**: "Learning to Select Knowledge for Response Generation in Dialog Systems". IJCAI(2019) [[paper]](https://www.ijcai.org/proceedings/2019/0706.paper) [[code-1]](https://github.com/siat-nlp/dialogue-models/tree/master/PostKS) [[code-2]](https://github.com/bzantium/Posterior-Knowledge-Selection) :star::star::star:
* **NKD**: "Knowledge Diffusion for Neural Dialogue Generation". ACL(2018) [[paper]](https://www.aclweb.org/anthology/P18-1138) [[data]](https://github.com/liushuman/neural-knowledge-diffusion) 
* **Dual Fusion**: "Smarter Response with Proactive Suggestion: A New Generative Neural Conversation Paradigm". IJCAI(2018) [[paper]](https://www.ijcai.org/proceedings/2018/0629.paper)
* **CCM**: "Commonsense Knowledge Aware Conversation Generation with Graph Attention". IJCAI(2018) [[paper]](https://www.ijcai.org/proceedings/2018/0643.paper) [[code-tf]](https://github.com/tuxchow/ccm) [[code-py]](https://github.com/Lyusungwon/CCM-pytorch)  :star::star::star::star::star:
* **MTask**: "A Knowledge-Grounded Neural Conversation Model". AAAI(2018)  [[paper]](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16710/16057)
* **GenDS**: "Flexible End-to-End Dialogue System for Knowledge Grounded Conversation". arXiv(2017) [[paper]](https://arxiv.org/paper/1709.04264.paper)

👆 [Back to Top](#paper-reading)


### Task-oriented Dialogue
* **P-ToD**: "Personalizing Task-oriented Dialog Systems via Zero-shot Generalizable Reward Function". CIKM(2022) [[paper]](https://dl.acm.org/doi/abs/10.1145/3511808.3557417)
* **Dialogic**: "Dialogic: Controllable Dialogue Simulation with In-Context Learning". EMNLP-Findings(2022) [[paper]](https://arxiv.org/abs/2210.04185) [[code]](https://github.com/Leezekun/dialogic) :star::star::star:
* **KB-Adapter**: "Injecting Domain Knowledge in Language Models for Task-Oriented Dialogue Systems". EMNLP(2022) [[paper]](https://arxiv.org/abs/2212.08120) [[code]](https://github.com/amazon-science/domain-knowledge-injection)
* **TacoBot**: "Bootstrapping a User-Centered Task-Oriented Dialogue System". Proceedings of Alexa Prize TaskBot(2021) [[paper]](https://arxiv.org/abs/2207.05223) :star::star::star:
* **USDA**: "User Satisfaction Estimation with Sequential Dialogue Act Modeling in Goal-oriented Conversational Systems". WWW(2022) [[paper]](https://arxiv.org/abs/2202.02912) [[code]](https://github.com/dengyang17/USDA)
* **USS**: "Simulating User Satisfaction for the Evaluation of Task-oriented Dialogue Systems". SIGIR(2021) [[paper]](https://arxiv.org/abs/2105.03748) [[data]](https://github.com/sunnweiwei/user-satisfaction-simulation)
* **NS-Dial**: "An Interpretable Neuro-Symbolic Reasoning Framework for Task-Oriented Dialogue Generation". ACL(2022) [[paper]](https://arxiv.org/abs/2203.05843) [[code]](https://github.com/shiquanyang/NS-Dial)
* **GALAXY**: "GALAXY: A Generative Pre-trained Model for Task-Oriented Dialog with Semi-Supervised Learning and Explicit Policy Injection". AAAI(2022) [[paper]](https://arxiv.org/abs/2111.14592) [[code]](https://github.com/siat-nlp/GALAXY)
* **PPTOD**: "Multi-Task Pre-Training for Plug-and-Play Task-Oriented Dialogue System". arXiv(2021) [[paper]](https://arxiv.org/abs/2109.14739) [[code]](https://github.com/awslabs/pptod)
* **ToDCL**: "Continual Learning in Task-Oriented Dialogue Systems". EMNLP(2021) [[paper]](https://aclanthology.org/2021.emnlp-main.590) [[code]](https://github.com/andreamad8/ToDCL)
* **IR-Net**: "Intention Reasoning Network for Multi-Domain End-to-end Task-Oriented Dialogue". EMNLP(2021) [[paper]](https://aclanthology.org/2021.emnlp-main.174)
* **HyKnow**: "HyKnow: End-to-End Task-Oriented Dialog Modeling with Hybrid Knowledge Management". ACL-Findings(2021) [[paper]](https://arxiv.org/abs/2105.06041) [[code]](https://github.com/truthless11/HyKnow)
* **DDMN**: "Dual Dynamic Memory Network for End-to-End Multi-turn Task-oriented Dialog Systems". COLING(2020) [[paper]](https://www.aclweb.org/anthology/2020.coling-main.362/) [[code]](https://github.com/siat-nlp/DDMN) :star::star::star:
* **ToD-BERT**: "ToD-BERT: Pre-trained Natural Language Understanding for Task-Oriented Dialogues". EMNLP(2020) [[paper]](https://www.aclweb.org/anthology/2020.emnlp-main.66) [[code]](https://github.com/jasonwu0731/ToD-BERT)
* **GraphDialog**: "GraphDialog: Integrating Graph Knowledge into End-to-End Task-Oriented Dialogue Systems". EMNLP(2020) [[paper]](https://www.aclweb.org/anthology/2020.emnlp-main.147) [[code]](https://github.com/shiquanyang/GraphDialog)
* **MARCO**: "Multi-Domain Dialogue Acts and Response Co-Generation". ACL(2020) [[paper]](https://arxiv.org/abs/2004.12363) [[code]](https://github.com/InitialBug/MarCo-Dialog)
* **DF-Net**: "Dynamic Fusion Network for Multi-Domain End-to-end Task-Oriented Dialog". ACL(2020) [[paper]](https://arxiv.org/paper/2004.11019.paper) [[code]](https://github.com/LooperXX/DF-Net)
* **MALA**: "MALA: Cross-Domain Dialogue Generation with Action Learning". AAAI(2020) [[paper]](https://arxiv.org/paper/1912.08442.paper)
* **SGD**: "Towards Scalable Multi-domain Conversational Agents: The Schema-Guided Dialogue Dataset". AAAI(2020) [[paper]](https://arxiv.org/abs/1909.05855) [[data]](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue)
* **CrossWOZ**: "CrossWOZ: A Large-Scale Chinese Cross-Domain Task-Oriented Dialogue Dataset". TACL(2020) [[paper]](https://arxiv.org/paper/2002.11893.paper) [[code]](https://github.com/thu-coai/CrossWOZ) 
* **MultiWOZ**: "MultiWOZ - A Large-Scale Multi-Domain Wizard-of-Oz Dataset for Task-Oriented Dialogue Modelling". EMNLP(2018) [[paper]](https://www.aclweb.org/anthology/D18-1547) [[code]](https://github.com/budzianowski/multiwoz)
* **Neural Task-Oriented Dialogue**: "Learning to Memorize in Neural Task-Oriented Dialogue Systems". MPhil Thesis(2019) [[paper]](https://arxiv.org/paper/1905.07687.paper) :star::star::star::star:
* **GLMP**: "Global-to-local Memory Pointer Networks for Task-Oriented Dialogue". ICLR(2019) [[paper]](https://arxiv.org/paper/1901.04713.paper) [[code]](https://github.com/jasonwu0731/GLMP) :star::star::star::star::star:
* **KB Retriever**: "Entity-Consistent End-to-end Task-Oriented Dialogue System with KB Retriever". EMNLP(2019) [[paper]](https://www.aclweb.org/anthology/D19-1013.paper) [[data]](https://github.com/yizhen20133868/Retriever-Dialogue)
* **TRADE**: "Transferable Multi-Domain State Generator for Task-Oriented
Dialogue Systems". ACL(2019) [[paper]](https://www.aclweb.org/anthology/P19-1078) [[code]](https://github.com/jasonwu0731/trade-dst)
* **WMM2Seq**: "A Working Memory Model for Task-oriented Dialog Response Generation". ACL(2019) [[paper]](https://www.aclweb.org/anthology/P19-1258)
* **Pretrain-Fine-tune**: "Training Neural Response Selection for Task-Oriented Dialogue Systems". ACL(2019) [[paper]](https://www.aclweb.org/anthology/P19-1536) [[data]](https://github.com/PolyAI-LDN/conversational-datasets)
* **Multi-level Mem**: "Multi-Level Memory for Task Oriented Dialogs". NAACL(2019) [[paper]](https://www.aclweb.org/anthology/N19-1375) [[code]](https://github.com/DineshRaghu/multi-level-memory-network)  :star::star::star:
* **BossNet**: "Disentangling Language and Knowledge in Task-Oriented Dialogs
". NAACL(2019) [[paper]](https://www.aclweb.org/anthology/N19-1126) [[code]](https://github.com/dair-iitd/BossNet)
* **SDN**: "Subgoal Discovery for Hierarchical Dialogue Policy Learning". EMNLP(2018) [[paper]](https://arxiv.org/abs/1804.07855) :star::star::star:
* **D3Q**: "Discriminative Deep Dyna-Q: Robust Planning for Dialogue Policy Learning". EMNLP(2018) [[paper]](https://arxiv.org/abs/1808.09442) [[code]](https://github.com/MiuLab/D3Q)
* **DDQ**: "Deep Dyna-Q: Integrating Planning for Task-Completion Dialogue Policy Learning". ACL(2018) [[paper]](https://aclweb.org/anthology/P18-1203) [[code]](https://github.com/MiuLab/DDQ)
* **MAD**: "Memory-augmented Dialogue Management for Task-oriented Dialogue Systems". TOIS(2018) [[paper]](https://arxiv.org/paper/1805.00150.paper)
* **TSCP**: "Sequicity: Simplifying Task-oriented Dialogue Systems with Single Sequence-to-Sequence Architectures". ACL(2018) [[paper]](https://www.aclweb.org/anthology/P18-1133) [[code]](https://github.com/WING-NUS/sequicity)
* **Mem2Seq**: "Mem2Seq: Effectively Incorporating Knowledge Bases into End-to-End Task-Oriented Dialog Systems". ACL(2018) [[paper]](https://www.aclweb.org/anthology/P18-1136) [[code]](https://github.com/HLTCHKUST/Mem2Seq) :star::star::star::star:
* **Topic-Seg-Label**: "A Weakly Supervised Method for Topic Segmentation and Labeling in Goal-oriented Dialogues via Reinforcement Learning". IJCAI(2018) [[paper]](https://www.ijcai.org/proceedings/2018/0612.paper) [[code]](https://github.com/truthless11/Topic-Seg-Label)
* **AliMe**: "AliMe Chat: A Sequence to Sequence and Rerank based Chatbot Engine". ACL(2017) [[paper]](https://aclweb.org/anthology/P17-2079)
* **KVR Net**: "Key-Value Retrieval Networks for Task-Oriented Dialogue". SIGDIAL(2017) [[paper]](https://www.aclweb.org/anthology/W17-5506) [[data]](https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/)

👆 [Back to Top](#paper-reading)


### Open-domain Dialogue
* **UniMC**: "UniMC: A Unified Framework for Long-Term Memory Conversation via Relevance Representation Learning". arXiv(2023) [[paper]](https://arxiv.org/abs/2306.10543)
* **Overview**: "Open-domain Dialogue Generation: What We Can Do, Cannot Do, And Should Do Next". ACL-NLP4ConvAI(2022) [[paper]](https://aclanthology.org/2022.nlp4convai-1.13) :star::star::star:
* **Multi-Session Chat**: "Beyond Goldfish Memory: Long-Term Open-Domain Conversation". ACL(2022) [[paper]](https://aclanthology.org/2022.acl-long.356) [[data]](https://parl.ai/projects/msc/) :star::star::star:
* **Chirpy Cardinal**: "Neural Generation Meets Real People: Building a Social, Informative Open-Domain Dialogue Agent". SIGDIAL(2022) [[paper]](https://arxiv.org/abs/2207.12021) [[code]](https://github.com/stanfordnlp/chirpycardinal) [[project]](https://stanfordnlp.github.io/chirpycardinal/)
* **TIL**: "Towards Efficient Dialogue Pre-training with Transferable and Interpretable Latent Structure". EMNLP(2022) [[paper]](https://arxiv.org/abs/2210.12461)
* **ProphetChat**: "ProphetChat: Enhancing Dialogue Generation with Simulation of Future Conversation". ACL(2022) [[paper]](https://aclanthology.org/2022.acl-long.68)
* **DialoFlow**: "Conversations Are Not Flat: Modeling the Dynamic Information Flow across Dialogue Utterances". ACL(2021) [[paper]](https://arxiv.org/abs/2106.02227) [[code]](https://github.com/ictnlp/DialoFlow)
* **DiSCoL**: "DiSCoL: Toward Engaging Dialogue Systems through Conversational Line Guided Response Generation". NAACL(2021) [[paper]](https://www.aclweb.org/anthology/2021.naacl-demos.4) [[code]](https://github.com/PlusLabNLP/Dialogue_System_Hackathon)
* **DialogBERT**: "DialogBERT: Discourse-Aware Response Generation via Learning to Recover and Rank Utterances". AAAI(2021) [[paper]](https://arxiv.org/paper/2012.01775.paper)
* **BlenderBot**: "Recipes for Building an Open-Domain Chatbot". EACL(2021) [[paper]](https://arxiv.org/abs/2004.13637) [[code]](https://huggingface.co/docs/transformers/model_doc/blenderbot)
* **CDial-GPT**: "A Large-Scale Chinese Short-Text Conversation Dataset". NLPCC(2020) [[paper]](https://arxiv.org/paper/2008.03946.paper) [[code]](https://github.com/thu-coai/CDial-GPT)
* **DialoGPT**: "DialoGPT : Large-Scale Generative Pre-training for Conversational Response Generation". ACL(2020) [[paper]](https://arxiv.org/paper/1911.00536.paper) [[code]](https://github.com/microsoft/DialoGPT) :star::star::star::star:
* **CG-Policy**: "Conversational Graph Grounded Policy Learning for Open-Domain Conversation Generation". ACL(2020) [[paper]](https://www.aclweb.org/anthology/2020.acl-main.166)
* **PLATO-XL**: "PLATO-XL: Exploring the Large-scale Pre-training of Dialogue Generation". arXiv(2021) [[paper]](https://arxiv.org/abs/2109.09519) [[code]](https://github.com/PaddlePaddle/Knover/tree/develop/projects)
* **PLATO-2**: "PLATO-2: Towards Building an Open-Domain Chatbot via Curriculum Learning". ACL-Findings(2021) [[paper]](https://arxiv.org/abs/2006.16779) [[code]](https://github.com/PaddlePaddle/Knover/tree/develop/projects/PLATO-2)
* **PLATO**: "PLATO: Pre-trained Dialogue Generation Model with Discrete Latent Variable". ACL(2020) [[paper]](https://arxiv.org/paper/1910.07931.paper) [[code]](https://github.com/PaddlePaddle/Research/tree/master/NLP/Dialogue-PLATO) 
* **Guyu**: "An Empirical Investigation of Pre-Trained Transformer Language Models for Open-Domain Dialogue Generation". arXiv(2020) [[paper]](https://arxiv.org/paper/2003.04195.paper) [[code]](https://github.com/lipiji/Guyu)
* **CL4Dialogue**: "Group-wise Contrastive Learning for Neural Dialogue Generation". EMNLP-Findings(2020) [[paper]](https://www.aclweb.org/anthology/2020.findings-emnlp.70) [[code]](https://github.com/hengyicai/ContrastiveLearning4Dialogue) :star::star::star:
* **Neg-train**: "Negative Training for Neural Dialogue Response Generation". ACL(2020) [[paper]](https://www.aclweb.org/anthology/2020.acl-main.185) [[code]](https://github.mit.edu/tianxing/negativetraining_acl2020)
* **HDSA**: "Semantically Conditioned Dialog Response Generation via Hierarchical Disentangled Self-Attention". ACL(2019) [[paper]](https://www.aclweb.org/anthology/P19-1360) [[code]](https://github.com/wenhuchen/HDSA-Dialog) :star::star::star:
* **CAS**: "Skeleton-to-Response: Dialogue Generation Guided by Retrieval Memory". NAACL(2019) [[paper]](https://www.aclweb.org/anthology/N19-1124) [[code]](https://github.com/jcyk/Skeleton-to-Response)
* **Edit-N-Rerank**: "Response Generation by Context-aware Prototype Editing". AAAI(2019) [[paper]](https://arxiv.org/paper/1806.07042.paper) [[code]](https://github.com/MarkWuNLP/ResponseEdit) :star::star::star:
* **HVMN**: "Hierarchical Variational Memory Network for Dialogue Generation". WWW(2018) [[paper]](https://dl.acm.org/citation.cfm?doid=3178876.3186077) [[code]](https://github.com/chenhongshen/HVMN)
* **XiaoIce**: "The Design and Implementation of XiaoIce, an Empathetic Social Chatbot". arXiv(2018) [[paper]](https://arxiv.org/paper/1812.08989.paper) :star::star::star:
* **D2A**: "Dialog-to-Action: Conversational Question Answering Over a Large-Scale Knowledge Base". NeurIPS(2018) [[paper]](https://papers.nips.cc/paper/7558-dialog-to-action-conversational-question-answering-over-a-large-scale-knowledge-base.paper) [[code]](https://github.com/guoday/Dialog-to-Action)
* **DAIM**: "Generating Informative and Diverse Conversational Responses via Adversarial Information Maximization". NeurIPS(2018) [[paper]](https://papers.nips.cc/paper/7452-generating-informative-and-diverse-conversational-responses-via-adversarial-information-maximization.paper)
* **REASON**: "Dialog Generation Using Multi-turn Reasoning Neural Networks". NAACL(2018) [[paper]](https://www.aclweb.org/anthology/N18-1186) 
* **STD/HTD**: "Learning to Ask Questions in Open-domain Conversational Systems with Typed Decoders". ACL(2018) [[paper]](https://www.aclweb.org/anthology/P18-1204) [[code]](https://github.com/victorywys/Learning2Ask_TypedDecoder) 
* **CSF**: "Generating Informative Responses with Controlled Sentence Function". ACL(2018) [[paper]](https://www.aclweb.org/anthology/P18-1139) [[code]](https://github.com/kepei1106/SentenceFunction)
* **DAWnet**: "Chat More: Deepening and Widening the Chatting Topic via A Deep Model". SIGIR(2018) [[paper]](https://dl.acm.org/citation.cfm?doid=3209978.3210061) [[code]](https://sigirdawnet.wixsite.com/dawnet)
* **ZSDG**: "Zero-Shot Dialog Generation with Cross-Domain Latent Actions". SIGDIAL(2018) [[paper]](https://www.aclweb.org/anthology/W18-5001) [[code]](https://github.com/snakeztc/NeuralDialog-ZSDG) 
* **DUA**: "Modeling Multi-turn Conversation with Deep Utterance Aggregation". COLING(2018) [[paper]](https://www.aclweb.org/anthology/C18-1317) [[code]](https://github.com/cooelf/DeepUtteranceAggregation)
* **Data-Aug**: "Sequence-to-Sequence Data Augmentation for Dialogue Language Understanding". COLING(2018) [[paper]](https://www.aclweb.org/anthology/C18-1105) [[code]](https://github.com/AtmaHou/Seq2SeqDataAugmentationForLU)
* **DC-MMI**: "Generating More Interesting Responses in Neural Conversation Models with Distributional Constraints". EMNLP(2018) [[paper]](https://www.aclweb.org/anthology/D18-1431) [[code]](https://github.com/abaheti95/DC-NeuralConversation)
* **cVAE-XGate/CGate**: "Better Conversations by Modeling, Filtering, and Optimizing for Coherence and Diversity". EMNLP(2018) [[paper]](https://www.aclweb.org/anthology/D18-1432) [[code]](https://github.com/XinnuoXu/CVAE_Dial)
* **Retrieval+multi-seq2seq**: "An Ensemble of Retrieval-Based and Generation-Based Human-Computer Conversation Systems". IJCAI(2018) [[paper]](https://www.ijcai.org/proceedings/2018/0609.paper)
* **DAM**: "Multi-Turn Response Selection for Chatbots with Deep Attention Matching Network". ACL(2018) [[paper]](https://www.aclweb.org/anthology/P18-1103) [[code]](https://github.com/baidu/Dialogue/tree/master/DAM) :star::star::star::star:
* **SMN**: "Sequential Matching Network: A New Architecture for Multi-turn Response Selection in Retrieval-Based Chatbots". ACL(2017) [[paper]](https://aclweb.org/anthology/P17-1046)  [[code]](https://github.com/MarkWuNLP/MultiTurnResponseSelection) :star::star::star:
* **CVAE/KgCVAE**: "Learning Discourse-level Diversity for Neural Dialog Models using Conditional Variational Autoencoders". ACL(2017) [[paper]](https://aclweb.org/anthology/P17-1061) [[code]](https://github.com/snakeztc/NeuralDialog-CVAE) :star::star::star:
* **TA-Seq2Seq**: "Topic Aware Neural Response Generation". AAAI(2017) [[paper]](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14563/14260) [[code]](https://github.com/LynetteXing1991/TA-Seq2Seq)
* **MA**: "Mechanism-Aware Neural Machine for Dialogue Response Generation". AAAI(2017) [[paper]](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14471/14267)
* **VHRED**: "A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues". AAAI(2017) [[paper]](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14567/14219) [[code]](https://github.com/julianser/hed-dlg-truncated)
* **HRED**: "Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models". AAAI(2016) [[paper]](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/11957/12160) [[code]](https://github.com/julianser/hed-dlg)
* **RL-Dialogue**: "Deep Reinforcement Learning for Dialogue Generation". EMNLP(2016) [[paper]](https://www.aclweb.org/anthology/D16-1127)
* **MMI**: "A Diversity-Promoting Objective Function for Neural Conversation Models". NAACL(2016)  [[paper]](https://www.aclweb.org/anthology/N16-1014) [[code]](https://github.com/jiweil/Neural-Dialogue-Generation)

👆 [Back to Top](#paper-reading)


### Dialogue Evaluation
* **BBScore**: "BBScore: A Brownian Bridge Based Metric for Assessing Text Coherence". AAAI(2024) [[paper]](https://arxiv.org/abs/2312.16893)
* **ChatEval**: "ChatEval: Towards Better LLM-based Evaluators through Multi-Agent Debate". arXiv(2023) [[paper]](https://arxiv.org/abs/2308.07201) [[code]](https://github.com/chanchimin/ChatEval)
* **ACCENT**: "ACCENT: An Automatic Event Commonsense Evaluation Metric for Open-Domain Dialogue Systems". ACL(2023) [[paper]](https://arxiv.org/abs/2305.07797) [[code]](https://github.com/PlusLabNLP/ACCENT)
* **GPTEval**: "GPTEval: NLG Evaluation using GPT-4 with Better Human Alignment". arXiv(2023) [[paper]](https://arxiv.org/abs/2303.16634)
* **GPTScore**: "GPTScore: Evaluate as You Desire". arXiv(2023) [[paper]](https://arxiv.org/abs/2302.04166) [[code]](https://github.com/jinlanfu/GPTScore)
* **LLMEval**: "Understanding the Effectiveness of Very Large Language Models on Dialog Evaluation". IWSDS(2023) [[paper]](https://arxiv.org/abs/2301.12004)
* **ChatEvalPlatform**: "Don't Forget Your ABC's: Evaluating the State-of-the-Art in Chat-Oriented Dialogue Systems". arXiv(2022) [[paper]](https://arxiv.org/abs/2212.09180) [[code]](https://github.com/emora-chat/ChatEvaluationPlatform)
* **MoralDial**: "MoralDial: A Framework to Train and Evaluate Moral Dialogue Systems via Constructing Moral Discussions". arXiv(2022) [[paper]](https://arxiv.org/abs/2212.10720)
* **MDD-Eval**: "MDD-Eval: Self-Training on Augmented Data for Multi-Domain Dialogue Evaluation". AAAI(2022) [[paper]](https://arxiv.org/abs/2112.07194) [[code]](https://github.com/e0397123/MDD-Eval)
* **Self-Eval**: "SelF-Eval: Self-supervised Fine-grained Dialogue Evaluation". COLING(2022) [[paper]](https://aclanthology.org/2022.coling-1.39/) [[code]](https://github.com/royny/self-eval)
* **FineD-Eval**: "FineD-Eval: Fine-grained Automatic Dialogue-Level Evaluation". EMNLP(2022) [[paper]](https://arxiv.org/abs/2210.13832) [[code]](https://github.com/e0397123/FineD-Eval)
* **FlowEval**: "FlowEval: A Consensus-Based Dialogue Evaluation Framework Using Segment Act Flows". EMNLP(2022) [[paper]](https://arxiv.org/abs/2202.06633)
* **IM2**: "IM^2: an Interpretable and Multi-category Integrated Metric Framework for Automatic Dialogue Evaluation". EMNLP(2022) [[paper]](https://preview.aclanthology.org/emnlp-22-ingestion/2022.emnlp-main.762/) [[code]](https://github.com/Jnunlplab/IM2)
* **RoMe**: "RoMe: A Robust Metric for Evaluating Natural Language Generation". ACL(2022) [[paper]](https://arxiv.org/abs/2203.09183) [[code]](https://github.com/rashad101/RoMe)
* **EAD**: "Rethinking and Refining the Distinct Metric". ACL(2022) [[paper]](https://arxiv.org/abs/2202.13587) [[code]](https://github.com/lsy641/Expectation-Adjusted-Distinct)
* **DiscoScore**: "DiscoScore: Evaluating Text Generation with BERT and Discourse Coherence". arXiv(2022) [[paper]](https://arxiv.org/abs/2201.11176) [[code]](https://github.com/AIPHES/DiscoScore)
* **CTC-Score**: "Compression, Transduction, and Creation: A Unified Framework for Evaluating Natural Language Generation". EMNLP(2021) [[paper]](https://arxiv.org/abs/2109.06379) [[code]](https://github.com/tanyuqian/ctc-gen-eval)
* **Q^2**: "$Q^{2}$: Evaluating Factual Consistency in Knowledge-Grounded Dialogues via Question Generation and Question Answering". EMNLP(2021) [[paper]](https://arxiv.org/abs/2104.08202) [[code]](https://github.com/orhonovich/q-squared)
* **QuantiDCE**: "Towards Quantifiable Dialogue Coherence Evaluation". ACL(2021) [[paper]](https://arxiv.org/abs/2106.00507) [[code]](https://github.com/James-Yip/QuantiDCE)
* **DynaEval**: "DynaEval: Unifying Turn and Dialogue Level Evaluation". ACL(2021) [[paper]](https://aclanthology.org/2021.acl-long.441) [[code]](https://github.com/e0397123/DynaEval)
* **Review**: "How to Evaluate Your Dialogue Models: A Review of Approaches". arXiv(2021) [[paper]](https://arxiv.org/abs/2108.01369)
* **ConvLabEval**: "Is Your Goal-Oriented Dialog Model Performing Really Well? Empirical Analysis of System-wise Evaluation". SIGDIAL(2020) [[paper]](https://arxiv.org/abs/2005.07362)
* **FED**: "Unsupervised Evaluation of Interactive Dialog with DialoGPT". SIGDIAL(2020) [[paper]](https://arxiv.org/abs/2006.12719) [[code]](https://github.com/Shikib/fed) [[data]](http://shikib.com/fed_data.json) :star::star::star:
* **Spot-the-Bot**: "Spot The Bot: A Robust and Efficient Framework for the Evaluation of Conversational Dialogue Systems". EMNLP(2020) [[paper]](https://www.aclweb.org/anthology/2020.emnlp-main.326) [[code]](https://github.com/jderiu/spot-the-bot-code)
* **CMADE**: "Beyond User Self-Reported Likert Scale Ratings: A Comparison Model for Automatic Dialog Evaluation". ACL(2020) [[paper]](https://aclanthology.org/2020.acl-main.126) [[code]](https://github.com/Weixin-Liang/dialog_evaluation_CMADE)
* **Coherence**: "Dialogue Coherence Assessment Without Explicit Dialogue Act Labels". ACL(2020) [[paper]](https://aclanthology.org/2020.acl-main.133) [[code]](https://github.com/UKPLab/acl2020-dialogue-coherence-assessment)
* **MAUDE**: "Learning an Unreferenced Metric for Online Dialogue Evaluation". ACL(2020) [[paper]](https://arxiv.org/abs/2005.00583) [[code]](https://github.com/facebookresearch/online_dialog_eval)
* **GRADE**: "GRADE: Automatic Graph-Enhanced Coherence Metric for Evaluating Open-Domain Dialogue Systems". ACL(2020) [[paper]](https://www.aclweb.org/anthology/2020.emnlp-main.742) [[code]](https://github.com/li3cmz/GRADE)
* **BLEURT**: "BLEURT: Learning Robust Metrics for Text Generation". ACL(2020) [[paper]](https://www.aclweb.org/anthology/2020.acl-main.704) [[code]](https://github.com/google-research/bleurt)
* **uBLEU**: "uBLEU: Uncertainty-Aware Automatic Evaluation Method for Open-Domain Dialogue Systems". ACL(2020) [[paper]](https://www.aclweb.org/anthology/2020.acl-srw.27) [[code]](https://github.com/YumaTsuta/upsilon_bleu)
* **USR**: "USR: An Unsupervised and Reference Free Evaluation Metric for Dialog Generation". ACL(2020) [[paper]](https://www.aclweb.org/anthology/2020.acl-main.64) [[code]](https://github.com/Shikib/usr)
* **ACUTE-EVAL**: "ACUTE-EVAL: Improved Dialogue Evaluation with Optimized Questions and Multi-turn Comparisons". NIPS ConvAI Workshop(2019) [[paper]](https://arxiv.org/abs/1909.03087) [[code]](https://github.com/facebookresearch/ParlAI/tree/main/parlai/crowdsourcing/tasks/acute_eval) :star::star::star:
* **InteractiveEval**: "Approximating Interactive Human Evaluation with Self-Play for Open-Domain Dialog Systems". NeurIPS(2019) [[paper]](https://proceedings.neurips.cc/paper/2019/file/fc9812127bf09c7bd29ad6723c683fb5-Paper.paper) [[code]](https://github.com/natashamjaques/neural_chat) :star::star::star:
* **ChatEval**: "ChatEval: A Tool for Chatbot Evaluation". NAACL(2019) [[paper]](https://aclanthology.org/N19-4011) [[project]](https://chateval.org/)
* **ADVMT**: "One `Ruler` for All Languages: Multi-Lingual Dialogue Evaluation with Adversarial Multi-Task Learning". IJCAI(2018) [[paper]](https://www.ijcai.org/proceedings/2018/0616.paper)


👆 [Back to Top](#paper-reading)



## Natural Language Generation

### Survey on NLG
* **CTG**: "A Survey of Controllable Text Generation using Transformer-based Pre-trained Language Models". arXiv(2022) [[paper]](https://arxiv.org/abs/2201.05337)
* **RTG**: "A Survey on Retrieval-Augmented Text Generation". arXiv(2022) [[paper]](https://arxiv.org/abs/2202.01110)
* **Hallucination**: "Survey of Hallucination in Natural Language Generation". arXiv(2022) [[paper]](https://arxiv.org/abs/2202.03629)
* **Evaluation**: "A Survey of Evaluation Metrics Used for NLG Systems". arXiv(2020) [[paper]](https://arxiv.org/abs/2008.12009)

👆 [Back to Top](#paper-reading)


### NLG Theories and Techniques
* **RED**: "Decoder-Only or Encoder-Decoder? Interpreting Language Model as a Regularized Encoder-Decoder". arXiv(2023) [[paper]](https://arxiv.org/abs/2304.04052) :star::star::star: 
* **LaMemo**: "LaMemo: Language Modeling with Look-Ahead Memory". NAACL(2022) [[paper]](https://arxiv.org/abs/2204.07341) [[code]](https://github.com/thu-coai/LaMemo)
* **PTG**: "Learning to Transfer Prompts for Text Generation". NAACL(2022) [[paper]](https://arxiv.org/abs/2205.01543) [[code]](https://github.com/RUCAIBox/Transfer-Prompts-for-Text-Generation)
* **EISL**: "Don't Take It Literally: An Edit-Invariant Sequence Loss for Text Generation". NAACL(2022) [[paper]](https://arxiv.org/abs/2106.15078) [[code]](https://github.com/guangyliu/EISL)
* **CT-Loss**: "A Simple Contrastive Learning Objective for Alleviating Neural Text Degeneration". arXiv(2022) [[paper]](https://arxiv.org/abs/2205.02517) [[code]](https://github.com/shaojiejiang/ct-loss)
* **SimCTG**: "A Contrastive Framework for Neural Text Generation". NeurIPS(2022) [[paper]](https://arxiv.org/abs/2202.06417) [[code]](https://github.com/yxuansu/simctg) :star::star::star:
* **CoNT**: "CoNT: Contrastive Neural Text Generation". NeurIPS(2022) [[paper]](https://arxiv.org/abs/2205.14690) [[code]](https://github.com/shark-nlp/cont)
* **Two-level-CL**: "Keywords and Instances: A Hierarchical Contrastive Learning Framework Unifying Hybrid Granularities for Text Generation". ACL(2022) [[paper]](https://aclanthology.org/2022.acl-long.304)
* **CLAPS**: "Contrastive Learning with Adversarial Perturbations for Conditional Text Generation". ICLR(2021) [[paper]](https://arxiv.org/abs/2012.07280) [[code]](https://github.com/seanie12/CLAPS) :star::star::star::star:
* **RetGen**: "RetGen: A Joint framework for Retrieval and Grounded Text Generation Modeling". AAAI(2022) [[paper]](https://arxiv.org/abs/2105.06597) [[code]](https://github.com/dreasysnail/RetGen)
* **RAG**: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks". NeurIPS(2020) [[paper]](https://arxiv.org/abs/2005.11401) [[code]](https://github.com/huggingface/transformers/blob/main/examples/research_projects/rag/README.md) :star::star::star::star:
* **TextGAIL**: "TextGAIL: Generative Adversarial Imitation Learning for Text Generation". AAAI(2021) [[paper]](https://arxiv.org/abs/2004.13796) [[code]](https://github.com/qywu/TextGAIL)
* **Latent-GLAT**: "*latent*-GLAT: Glancing at Latent Variables for Parallel Text Generation". ACL(2022) [[paper]](https://arxiv.org/abs/2204.02030) [[code]](https://github.com/baoy-nlp/Latent-GLAT)
* **s2s-ft**: "s2s-ft: Fine-Tuning Pretrained Transformer Encoders for Sequence-to-Sequence Learning". arXiv(2021) [[paper]](https://arxiv.org/abs/2110.13640) [[code]](https://github.com/microsoft/unilm/tree/master/s2s-ft)
* **EBM**: "Exposure Bias versus Self-Recovery: Are Distortions Really Incremental for Autoregressive Text Generation?". EMNLP(2021) [[paper]](https://aclanthology.org/2021.emnlp-main.415)
* **DiscoDVT**: "DiscoDVT: Generating Long Text with Discourse-Aware Discrete Variational Transformer". EMNLP(2021) [[paper]](https://arxiv.org/abs/2110.05999) [[code]](https://github.com/cdjhz/DiscoDVT)
* **DATG**: "Data Augmentation for Text Generation Without Any Augmented Data". ACL(2021) [[paper]](https://arxiv.org/abs/2105.13650)
* **JointGT**: "JointGT: Graph-Text Joint Representation Learning for Text Generation from Knowledge Graphs". ACL-Findings(2021) [[paper]](https://aclanthology.org/2021.findings-acl.223) [[code]](https://github.com/thu-coai/JointGT)
* **Embedding-Transfer**: "Bridging Subword Gaps in Pretrain-Finetune Paradigm for Natural Language Generation". ACL(2021) [[paper]](https://arxiv.org/abs/2106.06125) [[code]](https://github.com/DeepLearnXMU/embedding-transfer)
* **FastSeq**: "EL-Attention: Memory Efficient Lossless Attention for Generation". ICML(2021) [[paper]](https://arxiv.org/abs/2105.04779) [[code]](https://github.com/microsoft/fastseq) :star::star::star:
* **BERTSeq2Seq**: "Leveraging Pre-trained Checkpoints for Sequence Generation Tasks". TACL(2020) [[paper]](https://arxiv.org/paper/1907.12461.paper) [[code-tf]](https://github.com/google-research/google-research/tree/master/bertseq2seq) [[code-py]](https://github.com/huggingface/transformers) :star::star::star:
* **ERNIE-GEN**: "ERNIE-GEN: An Enhanced Multi-Flow Pre-training and Fine-tuning Framework for Natural Language Generation". IJCAI(2020) [[paper]](https://arxiv.org/paper/2001.11314.paper) [[code]](https://github.com/PaddlePaddle/ERNIE/tree/repro/ernie-gen) :star::star::star:
* **DITTO**: "Learning to Break the Loop: Analyzing and Mitigating Repetitions for Neural Text Generation". NeurIPS(2022) [[paper]](https://arxiv.org/abs/2206.02369) [[code]](https://github.com/Jxu-Thu/DITTO)
* **Repetition-Problem**: "A Theoretical Analysis of the Repetition Problem in Text Generation". AAAI(2021) [[paper]](https://arxiv.org/paper/2012.14660.paper) [[code]](https://github.com/fuzihaofzh/repetition-problem-nlg)
* **ENCONTER**: "ENCONTER: Entity Constrained Progressive Sequence Generation via Insertion-based Transformer". EACL(2021) [[paper]](https://arxiv.org/abs/2103.09548) [[code]](https://github.com/LARC-CMU-SMU/Enconter)
* **POINTER**: "POINTER: Constrained Progressive Text Generation via Insertion-based Generative Pre-training". EMNLP(2020) [[paper]](https://arxiv.org/abs/2005.00558) [[code]](https://github.com/dreasysnail/POINTER)
* **Cascaded Generation**: "Cascaded Text Generation with Markov Transformers". NeurIPS(2020) [[paper]](https://arxiv.org/paper/2006.01112.paper) [[code]](https://github.com/harvardnlp/cascaded-generation)
* **SFOT**: "Improving Text Generation with Student-Forcing Optimal Transport". EMNLP(2020) [[paper]](https://aclanthology.org/2020.emnlp-main.735)
* **OT-Seq2Seq**: "Improving Sequence-to-Sequence Learning via Optimal Transport". ICLR(2019) [[paper]](https://arxiv.org/abs/1901.06283) [[code]](https://github.com/LiqunChen0606/OT-Seq2Seq)

👆 [Back to Top](#paper-reading)



### Diffusion Models for NLG
* **RenderDiffusion**: "RenderDiffusion: Text Generation as Image Generation". arXiv(2023) [[paper]](https://arxiv.org/abs/2304.12519)
* **Masked-Diffusion-LM**: "A Cheaper and Better Diffusion Language Model with Soft-Masked Noise". arXiv(2023) [[paper]](https://arxiv.org/abs/2304.04746) [[code]](https://github.com/amazon-science/masked-diffusion-lm)
* **discrete-diffusion**: "A Reparameterized Discrete Diffusion Model for Text Generation". arXiv(2023) [[paper]](https://arxiv.org/abs/2302.05737) [[code]](https://github.com/hkunlp/reparam-discrete-diffusion)
* **Difformer**: "Difformer: Empowering Diffusion Models on the Embedding Space for Text Generation". arXiv(2023) [[paper]](https://arxiv.org/abs/2212.09412) :star::star::star:
* **GENIE**: "Text Generation with Diffusion Language Models: A Pre-training Approach with Continuous Paragraph Denoise". arXiv(2022) [[paper]](https://arxiv.org/abs/2212.11685) [[code]](https://github.com/microsoft/ProphetNet/tree/master/GENIE)
* **SED**: "Self-conditioned Embedding Diffusion for Text Generation". arXiv(2022) [[paper]](https://arxiv.org/abs/2211.04236)
* **SSD-LM**: "SSD-LM: Semi-autoregressive Simplex-based Diffusion Language Model for Text Generation and Modular Control". arXiv(2022) [[paper]](https://arxiv.org/abs/2210.17432) [[code]](https://github.com/xhan77/ssd-lm)
* **LD4LG**: "Latent Diffusion for Language Generation". arXiv(2022) [[paper]](https://arxiv.org/abs/2212.09462) [[code]](https://github.com/justinlovelace/latent-diffusion-for-language)
* **DiffusionBERT**: "DiffusionBERT: Improving Generative Masked Language Models with Diffusion Models". arXiv(2022) [[paper]](https://arxiv.org/abs/2211.15029) [[code]](https://github.com/Hzfinfdu/Diffusion-BERT)
* **DiffusER**: "DiffusER: Discrete Diffusion via Edit-based Reconstruction". arXiv(2022) [[paper]](https://arxiv.org/abs/2210.16886) [[code]](https://github.com/machelreid/diffuser)
* **SeqDiffuSeq**: "SeqDiffuSeq: Text Diffusion with Encoder-Decoder Transformers". arXiv(2022) [[paper]](https://arxiv.org/abs/2212.10325) [[code]](https://github.com/Yuanhy1997/SeqDiffuSeq)
* **DiffuSeq**: "DiffuSeq: Sequence to Sequence Text Generation with Diffusion Models". ICLR(2023) [[paper]](https://arxiv.org/abs/2210.08933) [[code]](https://github.com/Shark-NLP/DiffuSeq)
* **Diffusion-LM**: "Diffusion-LM Improves Controllable Text Generation". NeurIPS(2022) [[paper]](https://arxiv.org/abs/2205.14217) [[code]](https://github.com/XiangLi1999/Diffusion-LM) :star::star::star:
* **D3PM**: "Structured Denoising Diffusion Models in Discrete State-Spaces". NeurIPS(2021) [[paper]](https://arxiv.org/abs/2107.03006) [[code]](https://github.com/google-research/google-research/tree/master/d3pm)

👆 [Back to Top](#paper-reading)


### Controllable Generation
* **GeLaTo**: "Tractable Control for Autoregressive Language Generation". arXiv(2023) [[paper]](https://arxiv.org/abs/2304.07438)
* **Cognac**: "Controllable Text Generation with Language Constraints". arXiv(2022) [[paper]](https://arxiv.org/abs/2212.10466) [[code]](https://github.com/princeton-nlp/Cognac)
* **CriticControl**: "Critic-Guided Decoding for Controlled Text Generation". arXiv(2022) [[paper]](https://arxiv.org/abs/2212.10938)
* **LatentOps**: "Composable Text Controls in Latent Space with ODEs". arXiv(2022) [[paper]](https://arxiv.org/abs/2208.00638) [[code]](https://github.com/guangyliu/LatentOps)
* **FAST**: "FAST: Improving Controllability for Text Generation with Feedback Aware Self-Training". arXiv(2022) [[paper]](https://arxiv.org/abs/2210.03167)
* **DisCup**: "DisCup: Discriminator Cooperative Unlikelihood Prompt-tuning for Controllable Text Generation". EMNLP(2022) [[paper]](https://arxiv.org/abs/2210.09551) [[code]](https://github.com/littlehacker26/discriminator-cooperative-unlikelihood-prompt-tuning)
* **MultiControl**: "A Distributional Lens for Multi-Aspect Controllable Text Generation". EMNLP(2022) [[paper]](https://arxiv.org/abs/2210.02889) [[code]](https://github.com/HappyGu0524/MultiControl)
* **NADO**: "Controllable Text Generation with Neurally-Decomposed Oracle". NeurIPS(2022) [[paper]](https://arxiv.org/abs/2205.14219) [[code]](https://github.com/mtsomethree/constrdecoding)
* **Mix-Match**: "Mix and Match: Learning-free Controllable Text Generation using Energy Language Models". ACL(2022) [[paper]](https://aclanthology.org/2022.acl-long.31) [[code]](https://github.com/mireshghallah/mixmatch)
* **ControlPrefix**: "Controllable Natural Language Generation with Contrastive Prefixes". ACL-Findings(2022) [[paper]](https://aclanthology.org/2022.findings-acl.229)
* **MUCOCO**: "Controlled Text Generation as Continuous Optimization with Multiple Constraints". NeurIPS(2021) [[paper]](https://proceedings.neurips.cc/paper/2021/file/79ec2a4246feb2126ecf43c4a4418002-Paper.paper) [[code]](https://github.com/Sachin19/mucoco)
* **DExperts**: "DExperts: Decoding-Time Controlled Text Generation with Experts and Anti-Experts". ACL(2021) [[paper]](https://aclanthology.org/2021.acl-long.522) [[code]](https://github.com/alisawuffles/DExperts)
* **FUDGE**: "FUDGE: Controlled Text Generation With Future Discriminators". NAACL(2021) [[paper]](https://arxiv.org/abs/2104.05218) [[code]](https://github.com/yangkevin2/naacl-2021-fudge-controlled-generation)
* **GeDi**: "GeDi: Generative Discriminator Guided Sequence Generation". EMNLP-Findings(2021) [[paper]](https://aclanthology.org/2021.findings-emnlp.424/) [[code]](https://github.com/salesforce/GeDi)
* **GDC**: "A Distributional Approach to Controlled Text Generation". ICLR(2021) [[paper]](https://arxiv.org/abs/2012.11635) [[code]](https://github.com/naver/gdc) :star::star::star:
* **CoCon**: "CoCon: A Self-Supervised Approach for Controlled Text Generation". ICLR(2021) [[paper]](https://arxiv.org/abs/2006.03535) [[code]](https://github.com/alvinchangw/COCON_ICLR2021)
* **PPLM**: "Plug and Play Language Models: A Simple Approach to Controlled Text Generation". ICLR(2020) [[paper]](https://arxiv.org/abs/1912.02164) [[code]](https://github.com/uber-research/PPLM) :star::star::star:
* **CTRL**: "CTRL: A Conditional Transformer Language Model for Controllable Generation". arXiv(2019) [[paper]](https://arxiv.org/abs/1909.05858) [[code]](https://github.com/salesforce/ctrl)

👆 [Back to Top](#paper-reading)


### Text Planning
* **CoScript**: "Distilling Script Knowledge from Large Language Models for Constrained Language Planning". ACL(2023) [[paper]](https://arxiv.org/abs/2305.05252) [[code]](https://github.com/siyuyuan/coscript)
* **RSTGen**: "RSTGen: Imbuing Fine-Grained Interpretable Control into Long-FormText Generators". NAACL(2022) [[paper]](https://arxiv.org/abs/2205.12590)
* **Time Control**: "Language Modeling via Stochastic Processes". ICLR(2022) [[paper]](https://arxiv.org/abs/2203.11370) [[code]](https://github.com/rosewang2008/language_modeling_via_stochastic_processes) :star::star::star::star::star:
* **PLANET**: "PLANET: Dynamic Content Planning in Autoregressive Transformers for Long-form Text Generation". ACL(2022) [[paper]](https://arxiv.org/abs/2203.09100)
* **EventPlan**: "Event Transition Planning for Open-ended Text Generation". ACL-Findings(2022) [[paper]](https://arxiv.org/abs/2204.09453) [[code]](https://github.com/qtli/EventPlanforTextGen)
* **CETP**: "Knowledge-based Review Generation by Coherence Enhanced Text Planning". SIGIR(2021) [[paper]](https://dl.acm.org/doi/10.1145/3404835.3462865) :star::star::star:
* **PlanGen**: "Plan-then-Generate: Controlled Data-to-Text Generation via Planning". EMNLP-Findings(2021) [[paper]](https://aclanthology.org/2021.findings-emnlp.76) [[code]](https://github.com/yxuansu/PlanGen)
* **DYPLOC**: "DYPLOC: Dynamic Planning of Content Using Mixed Language Models for Text Generation". ACL(2021) [[paper]](https://arxiv.org/abs/2106.00791) [[code]](https://github.com/XinyuHua/dyploc-acl2021)
* **Tree-PLAN**: "Infobox-to-text Generation with Tree-like Planning based Attention Network". IJCAI(2020) [[paper]](https://www.ijcai.org/proceedings/2020/522)
* **ProphetNet**: "ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training". EMNLP-Findings(2020) [[paper]](https://arxiv.org/abs/2001.04063) [[code]](https://github.com/microsoft/ProphetNet) :star::star::star:
* **PAIR**: "PAIR: Planning and Iterative Refinement in Pre-trained Transformers for Long Text Generation". EMNLP(2020) [[paper]](https://aclanthology.org/2020.emnlp-main.57) [[code]](https://github.com/XinyuHua/pair-emnlp2020)
* **SentPlan**: "Sentence-Level Content Planning and Style Specification for Neural Text Generation". EMNLP(2019) [[paper]](https://aclanthology.org/D19-1055) [[code]](https://github.com/XinyuHua/textgen-emnlp19)
* **PHVM**: "Long and Diverse Text Generation with Planning-based Hierarchical Variational Model". EMNLP(2019) [[paper]](https://www.aclweb.org/anthology/D19-1321) [[code]](https://github.com/ZhihongShao/Planning-based-Hierarchical-Variational-Model)
* **TwinNet**: "Twin Networks: Matching the Future for Sequence Generation". ICLR(2018) [[paper]](https://arxiv.org/abs/1708.06742) [[code]](https://github.com/dmitriy-serdyuk/twin-net)
* **PAG**: "Plan, Attend, Generate: Planning for Sequence-to-Sequence Models". NIPS(2017) [[paper]](https://proceedings.neurips.cc/paper/2017/file/b030afbb3a8af8fb0759241c97466ee4-Paper.paper)

👆 [Back to Top](#paper-reading)


### Decoding Algorithms
* **Lookahead Decoding**: "Breaking the Sequential Dependency of LLM Inference Using Lookahead Decoding". LMSYS Org(2023) [[Blog]](https://lmsys.org/blog/2023-11-21-lookahead-decoding/) [[code]](https://github.com/hao-ai-lab/LookaheadDecoding) :star::star::star:
* **Speculative Sampling**: "Accelerating Large Language Model Decoding with Speculative Sampling". arXiv(2023) [[paper]](http://arxiv.org/abs/2302.01318)
* **Speculative Decoding**: "Fast Inference from Transformers via Speculative Decoding". ICML(2023) [[paper]](http://arxiv.org/abs/2211.17192) [[code]](https://github.com/OptimalScale/LMFlow/blob/main/scripts/speculative_decoding/README.md)
* **Parallel Decoding**: "Accelerating Transformer Inference for Translation via Parallel Decoding". ACL(2023) [[paper]](https://arxiv.org/abs/2305.10427) [[code]](https://github.com/teelinsan/parallel-decoding)
* **EAD**: "The Stable Entropy Hypothesis and Entropy-Aware Decoding: An Analysis and Algorithm for Robust Natural Language Generation". arXiv(2023) [[paper]](https://arxiv.org/abs/2302.06784) [[code]](https://github.com/kushalarora/transformers/blob/main/src/transformers/generation_utils.py#L1894)
* **Contrastive Search**: "Contrastive Search Is What You Need For Neural Text Generation". TMLR(2023) [[paper]](https://arxiv.org/abs/2210.14140) [[code]](https://github.com/yxuansu/Contrastive_Search_Is_What_You_Need) [[blog]](https://huggingface.co/blog/introducing-csearch) :star::star::star:
* **Momentum Decoding**: "Momentum Decoding: Open-ended Text Generation As Graph Exploration". arXiv(2022) [[paper]](https://arxiv.org/abs/2212.02175) [[code]](https://github.com/gmftbyGMFTBY/MomentumDecoding)
* **Crowd Sampling**: "Follow the Wisdom of the Crowd: Effective Text Generation via Minimum Bayes Risk Decoding". arXiv(2022) [[paper]](https://arxiv.org/abs/2211.07634) [[code]](https://github.com/suzgunmirac/crowd-sampling)
* **RankGen**: "RankGen: Improving Text Generation with Large Ranking Models". EMNLP(2022) [[paper]](https://arxiv.org/abs/2205.09726) [[code]](https://github.com/martiansideofthemoon/rankgen)
* **Contrastive Decoding**: "Contrastive Decoding: Open-ended Text Generation as Optimization". arXiv(2022) [[paper]](https://arxiv.org/abs/2210.15097) [[code]](https://github.com/xiangli1999/contrastivedecoding)
* **COLD**: "COLD Decoding: Energy-based Constrained Text Generation with Langevin Dynamics". NeurIPS(2022) [[paper]](https://arxiv.org/abs/2202.11705) [[code]](https://github.com/qkaren/COLD_decoding) :star::star::star:
* **Lattice**: "Massive-scale Decoding for Text Generation using Lattices". NAACL(2022) [[paper]](https://arxiv.org/abs/2112.07660) [[code]](https://github.com/jiacheng-xu/lattice-generation)
* **KID**: "Knowledge Infused Decoding". ICLR(2022) [[paper]](https://arxiv.org/abs/2204.03084) [[code]](https://github.com/microsoft/kid)
* **NeuroLogic A*esque**: "NeuroLogic A *esque Decoding: Constrained Text Generation with Lookahead Heuristics". NAACL(2022) [[paper]](https://arxiv.org/abs/2112.08726) [[code]](https://github.com/GXimingLu/a_star_neurologic)
* **NeuroLogic**: "NeuroLogic Decoding: (Un)supervised Neural Text Generation with Predicate Logic Constraints". NAACL(2021) [[paper]](https://aclanthology.org/2021.naacl-main.339) [[code]](https://github.com/GXimingLu/neurologic_decoding)
* **DeLorean**: "Back to the Future: Unsupervised Backprop-based Decoding for Counterfactual and Abductive Commonsense Reasoning". EMNLP(2020) [[paper]](https://aclanthology.org/2020.emnlp-main.58) [[code]](https://github.com/qkaren/unsup_gen_for_cms_reasoning)
* **Top-p (Nucleus) Sampling**: "The Curious Case of Neural Text Degeneration". ICLR(2020) [[paper]](https://openreview.net/forum?id=rygGQyrFvH) [[code]](https://github.com/ari-holtzman/degen)
* **BP Decoding**: "Blockwise Parallel Decoding for Deep Autoregressive Models". NIPS(2018) [[paper]](http://arxiv.org/abs/1811.03115)
* **Disjunctive Constraints**: "Guided Generation of Cause and Effect". IJCAI(2020) [[paper]](https://arxiv.org/abs/2107.09846) [[code-huggingface]](https://huggingface.co/blog/constrained-beam-search)
* **CGMH**: "CGMH: Constrained Sentence Generation by Metropolis-Hastings Sampling". AAAI(2019) [[paper]](https://arxiv.org/abs/1811.10996) [[code]](https://github.com/NingMiao/CGMH)
* **DBS**: "Directed Beam Search: Plug-and-Play Lexically Constrained Language Generation". arXiv(2020) [[paper]](https://arxiv.org/abs/2012.15416) [[code]](https://github.com/dapascual/DirectedBeamSearch)
* **DBA**: "Fast Lexically Constrained Decoding with Dynamic Beam Allocation for Neural Machine Translation". NAACL(2018) [[paper]](https://aclanthology.org/N18-1119) [[code-official]](https://github.com/awslabs/sockeye) [[code-fairseq]](https://github.com/facebookresearch/fairseq/blob/main/examples/constrained_decoding/README.md)
* **GBS**: "Lexically Constrained Decoding for Sequence Generation Using Grid Beam Search". ACL(2017) [[paper]](https://aclanthology.org/P17-1141) [[code]](https://github.com/chrishokamp/constrained_decoding)

👆 [Back to Top](#paper-reading)
