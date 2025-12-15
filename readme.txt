├── feature_data/          # Directory for storing traffic features
│   └── traffic_feature.csv # Statistical traffic features extracted after preprocessing
├── json_data/             # Directory for storing JSON-formatted traffic data (output of PCAP processing)
├── model/                 # Directory for storing trained model files
├── pcap_data/             # Directory for storing raw network traffic data (PCAP format)
├── get_data_split_train_test.py # Script: Extract traffic fingerprints, generate cross-region labels, split train/test sets
├── index.csv              # Index file: Maps PCAP files in pcap_data to IP addresses and IP attributes
├── pcap_preprocess.py     # Preprocessing script: Convert PCAP to JSON (saved in json_data) + extract statistical features (saved in feature_data)
├── TrafficCL.py           # Model script: Contains model training and testing logic
└── readme                 # Project documentation