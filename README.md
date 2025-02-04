# TOBIGS_GBTI


![PUBG](https://github.com/user-attachments/assets/a5958898-fc90-488c-b840-fd079fa239c6)


## 프로젝트 구조 (Architecture)

```plaintext
project-root/
├── analysis
│   ├── collector.ipynb
│   ├── group.ipynb
│   ├── killer.ipynb
│   ├── manner.ipynb
│   └── social.ipynb
├── confer
│   ├── inference
│   │   ├── collector_infer.py
│   │   ├── explorer_infer.py
│   │   ├── killer_infer.py
│   │   ├── manner_infer.py
│   │   └── social_infer.py
│   └── model
│       ├── collector
│       │   └── scaler.pkl
│       ├── explorer
│       │   └── explorer_model.pkl
│       ├── killer
│       │   ├── scaler.pkl
│       │   └── svm_model.pkl
│       ├── manner
│       │   ├── scaler.pkl
│       │   ├── svm_model.pkl
│       │   └── thresholds.pkl
│       └── social
│           ├── group_model_svm.pkl
│           └── group_scaler.pkl
├── all_infer.py
├── check.ipynb
├── data_inmemory.py
├── pubg_fetch.py
├── requirements.txt
├── run.py
├── README.md
└── data_crawling.py
