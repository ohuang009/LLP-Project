# WaterRAG full-paper accepted nodes

- Run: `waterrag_full_node_pipeline_chunked_20260805T101654Z`
- Scope: `full_paper` (282 narrative sentences)
- Candidate generation: 714 grammatical triples; 995 unique subject/object arguments
- Final result: 181 accepted mentions consolidated into 46 canonical nodes
- DeepSeek: `deepseek-r1:7b`; used = `true`
- Traceability: 181/181 accepted mentions have exact evidence

| Node type | Canonical node | Mentions | Supporting sentence IDs |
|---|---|---:|---|
| Agent | three agent components | 1 | `sent_2b54b5d215d9` |
| AISystem | Complete reranking prompt | 1 | `sent_8ec75625ce40` |
| AISystem | fine-tuning, retrieval-augmented generation (RAG) frameworks | 1 | `sent_5d127ef9d33f` |
| AISystem | Multiagent System | 1 | `sent_57038af05de9` |
| AISystem | Naive RAG | 1 | `sent_a9feec35bd8f` |
| AISystem | WaterRAG | 99 | `sent_00b23eac0f2e`, `sent_01a37f412b30`, `sent_03ed1dc8bf00`, `sent_0b89fd8e4b2d`, `sent_0cda6c34874f`, `sent_0d90702661b5`, `sent_0fa48393c1c5`, `sent_11f5926ea0a2`, `sent_1942effe2117`, `sent_19bc62e029d6`, `sent_215780cdae77`, `sent_22bc3d69ec11`, `sent_28b10806ba3d`, `sent_296479ff62d5`, `sent_2a00de3f85ac`, `sent_2b54b5d215d9`, `sent_2fa461c899a4`, `sent_3070c6958533`, `sent_341fa972060f`, `sent_3473ffc76f1b`, `sent_37505fbb07cb`, `sent_3874aad995b1`, `sent_3c4c1f44e3c1`, `sent_43b5a441f23d`, `sent_4570ae9e76a1`, `sent_4b03ae08167e`, `sent_4b80bca57e10`, `sent_4d6b477ca205`, `sent_57038af05de9`, `sent_5b0617970179`, `sent_5b908bd2e991`, `sent_5d83958bacf1`, `sent_5db03a4f4818`, `sent_611023653917`, `sent_64d260f19bfb`, `sent_68473f4536e2`, `sent_732bfe4ede87`, `sent_7414190cc892`, `sent_7545c75f1b27`, `sent_764c763281d6`, `sent_7a8ab042a975`, `sent_7fa7d19d0ed3`, `sent_80cb8cafd1aa`, `sent_833d8a805a0b`, `sent_85c599ade355`, `sent_899e58d65cf5`, `sent_99563181320d`, `sent_9b9c1e8710cb`, `sent_a36c4325d689`, `sent_a9feec35bd8f`, `sent_aa3dbd40a89b`, `sent_abf14ddef742`, `sent_ae318cbaaaa2`, `sent_af6f423e3486`, `sent_b009800c705c`, `sent_b13d544a89f8`, `sent_b1ca1dcd8312`, `sent_b446261dd3dc`, `sent_b4981e9d6944`, `sent_b5045eead9af`, `sent_b508075d9499`, `sent_b88f994ee239`, `sent_ba8510be6a70`, `sent_bdb1cff51bc3`, `sent_bf68fe8da3d6`, `sent_c3922803b106`, `sent_c3ac86fccc94`, `sent_c4f30bfb7da1`, `sent_c9e61e612a7a`, `sent_cb6d13ed01d3`, `sent_cfb013c734cf`, `sent_d066816235b9`, `sent_d0cb4bd29412`, `sent_d4aff4af8b07`, `sent_d4c7ed97d242`, `sent_d926260d9f03`, `sent_e2fd4e56c2b1`, `sent_e3506b35f87a`, `sent_e362af007a87`, `sent_e374268ff566`, `sent_e77603e485ee`, `sent_e9022c6102a6`, `sent_e91e84850d6a`, `sent_e9dd7c0ec31b`, `sent_eb1bc1ad9dd9`, `sent_eb996e5b9edf`, `sent_eba599adce9d`, `sent_ef9122879d34`, `sent_f1091d968d7b`, `sent_f53e46ac54a1`, `sent_f8f5ef4443b6`, `sent_f9fdb98a2a98` |
| Challenge | incorrect categorization | 1 | `sent_76b155cca747` |
| Claim | addition | 1 | `sent_f7f64f3b90e3` |
| Claim | All comparisons | 1 | `sent_c594b96fca2e` |
| Claim | correct answer | 1 | `sent_3874aad995b1` |
| Claim | larger scale | 1 | `sent_91f9e9eb8f3f` |
| Claim | meaningful gains | 1 | `sent_ae318cbaaaa2` |
| Claim | relatively smaller improvement margin | 1 | `sent_91f9e9eb8f3f` |
| Contaminant | ammonia | 3 | `sent_d73fd460c13e`, `sent_e9022c6102a6`, `sent_f7b46f83784b` |
| Contaminant | nitrate | 1 | `sent_e2fd4e56c2b1` |
| Dataset | the most relevant chunks | 1 | `sent_a36c4325d689` |
| DataSource | only 20 cited references | 1 | `sent_76b155cca747` |
| EnvironmentalProcess | achieving net-zero carbon emissions in wastewater treatment | 1 | `sent_7b779561a8ee` |
| Experiment | systematic ablation experiments | 1 | `sent_f1091d968d7b` |
| Metric | energy consumption | 2 | `sent_70537cf50c57`, `sent_c70b81bbb9c7` |
| Model | GPT-4 | 2 | `sent_080120e18e48`, `sent_99a92b54c5ad` |
| Model | GPT-4.1 | 27 | `sent_11f5926ea0a2`, `sent_1753a2d1cfb9`, `sent_215780cdae77`, `sent_27084f07c230`, `sent_43b5a441f23d`, `sent_4ca678ec6abd`, `sent_5d83958bacf1`, `sent_64c02adaa6c2`, `sent_8de7de6cad10`, `sent_a4235fce8890`, `sent_b446261dd3dc`, `sent_b8e9556e73da`, `sent_bc32517049a1`, `sent_bf68fe8da3d6`, `sent_c2ffad76fc46`, `sent_c3ac86fccc94`, `sent_d1c0947a3504`, `sent_e13acdaef9e5`, `sent_e2fd4e56c2b1`, `sent_e362af007a87`, `sent_e374268ff566`, `sent_e9022c6102a6`, `sent_e91e84850d6a`, `sent_eb996e5b9edf`, `sent_f1683bda91ef` |
| Model | Llama 3.1 | 7 | `sent_27084f07c230`, `sent_4ca678ec6abd`, `sent_6914768eb9bf`, `sent_ae318cbaaaa2`, `sent_e374268ff566`, `sent_e91e84850d6a`, `sent_ed91a717a6bd` |
| Model | Random Forest | 1 | `sent_7c313479a5a7` |
| Model | the baseline model | 1 | `sent_6b1e1fa62bc1` |
| Person | Bing-Jie Ni | 1 | `metadata-sentence_b81787f254b3ec961dc7` |
| Person | David Waite | 1 | `metadata-sentence_b81787f254b3ec961dc7` |
| Person | Haoran Duan | 1 | `metadata-sentence_b81787f254b3ec961dc7` |
| Person | Jiaying Li | 1 | `metadata-sentence_b81787f254b3ec961dc7` |
| Person | Mudi Zhai | 1 | `metadata-sentence_b81787f254b3ec961dc7` |
| Person | Qingyun Zeng | 1 | `metadata-sentence_b81787f254b3ec961dc7` |
| Person | Qixiang Zhu | 1 | `metadata-sentence_b81787f254b3ec961dc7` |
| Person | Ruihong Qiu | 1 | `metadata-sentence_b81787f254b3ec961dc7` |
| Policy | Policymakers | 1 | `sent_1efa2eceb0b3` |
| Publication | 11 engineering references | 1 | `sent_00b23eac0f2e` |
| Publication | WaterRAG: A Multiagent Retrieval-Augmented Generation Framework to Support Water Industry Transitions to Net-Zero | 1 | `metadata-sentence_adf13224d2c125c73b2c` |
| ScientificConcept | more comprehensive and relevant retrieved information | 1 | `sent_b009800c705c` |
| SoftwareArtifact | Deepeval framework | 1 | `sent_7c313479a5a7` |
| SoftwareArtifact | open-source Llama3 − 8B model | 1 | `sent_50e65d13a8ed` |
| Task | 10 review tasks | 1 | `sent_e9dd7c0ec31b` |
| Technology | fine-tuning techniques | 1 | `sent_fcb8005cb33f` |
| Tool | Llama-3.1 − 8B | 1 | `sent_e91e84850d6a` |
| Tool | structured decision-support tool | 1 | `sent_b508075d9499` |
| TreatmentProcess | activated sludge process | 2 | `sent_6914768eb9bf` |
| TreatmentProcess | chlorination | 1 | `sent_3c173676ea1f` |
| WaterSystem | domain-specific wastewater treatment tasks | 1 | `sent_1942effe2117` |

## Counts by node type

- AISystem: 5
- Agent: 1
- Challenge: 1
- Claim: 6
- Contaminant: 2
- DataSource: 1
- Dataset: 1
- EnvironmentalProcess: 1
- Experiment: 1
- Metric: 1
- Model: 5
- Person: 8
- Policy: 1
- Publication: 2
- ScientificConcept: 1
- SoftwareArtifact: 2
- Task: 1
- Technology: 1
- Tool: 2
- TreatmentProcess: 2
- WaterSystem: 1
