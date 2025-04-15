# ADKF
A long-term forcaste model based on KAN network

![Repository Creation Interface](repo-creation-screenshot.png)  
*Figure: GitHub repository creation interface for reference*

## 📌 Project Overview
A long-term forecasting model based on Kernel Adaptive Network (KAN), designed for:
- Financial time series prediction
- Meteorological data modeling
- Industrial equipment lifespan forecasting

## 🛠️ Environment Configuration

Configuration
dependencies:
  python: ">=3.9"
  pytorch: "2.1.0"
  cuda: "11.8/12.1"
Other essential_packages in requirements.txt

🚀 Quick Start

Training
bash ADKF-main/configs/ADKF.sh --is_training 1

Test
bash ADKF-main/configs/ADKF.sh --is_training 0

🌟 Key Features
Feature	Advantage
Adaptive Kernels	Dynamic bandwidth adjustment for non-stationary series
Multi-scale Modeling	Captures both short-term and long-term patterns
Memory Compression	40% reduction in GPU usage via Temporal Attention
