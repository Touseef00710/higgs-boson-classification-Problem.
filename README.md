##  Project Description  
This repository implements a **deep learning model** to detect Higgs boson signals in particle collision data from CERN's LHC experiments. The system:  

- Processes **30 anonymized physics features** using a custom TensorFlow/Keras pipeline  
- Uses a **3-layer neural network** with dropout/BatchNorm for regularization  
- Outperforms XGBoost baselines by **+3.1% accuracy** (81.3% test accuracy)  
- Includes **automated training workflows** with EarlyStopping and LR scheduling  

# Key applications:  
1: Particle physics research  
2: Benchmark for ML in high-energy physics  
3: Case study in imbalanced binary classification  

## Dataset: Derived from the [Kaggle Higgs Boson Challenge](https://www.kaggle.com/c/higgs-boson)  
