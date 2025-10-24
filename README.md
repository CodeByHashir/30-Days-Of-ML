# 30 Days of Machine Learning - Fast-Track Course

> **A modern, hands-on journey through Machine Learning from core foundations to advanced deep learning and MLOps, with cutting-edge methods and real-world projects.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![ML](https://img.shields.io/badge/ML-Scikit--learn%20%7C%20PyTorch%20%7C%20HuggingFace-orange.svg)](https://scikit-learn.org/)

## Course Overview

This intensive 30-day program is designed for engineers who want to master modern Machine Learning without getting bogged down in outdated concepts. We focus on **practical application**, **industry best practices**, and **cutting-edge tools** that are actually used in production today.

### What Makes This Course Different

- **Fast-Track Approach**: Skip hand derivations, focus on intuition + application
- **Modern Stack**: PyTorch, HuggingFace, MLflow, Docker, FastAPI
- **Real Datasets**: Kaggle, UCI, HuggingFace Datasets (no toy examples)
- **End-to-End Projects**: Complete pipelines with deployment
- **Industry-Ready**: MLOps, model deployment, and production best practices

## Curriculum Structure

### Week 1: Core Foundations
**Goal**: Build intuition + hands-on workflow

| Day | Topic | Key Technologies |
|-----|-------|------------------|
| 1 | Python for DS Refresher | NumPy, Pandas, Matplotlib, Seaborn |
| 2 | Data Preprocessing | Missing values, scaling, encoding |
| 3 | EDA Best Practices | Feature exploration, visualization, correlations |
| 4 | ML Pipeline (Scikit-learn) | Train/test split, metrics |
| 5 | Linear & Logistic Regression | Intuition + applications |
| 6 | Decision Trees & Random Forests | Feature importance, interpretability |
| 7 | **Mini-Project**: Churn Prediction | End-to-end pipeline |

### Week 2: Essential ML Algorithms
**Goal**: Cover key classical + ensemble ML methods

| Day | Topic | Key Technologies |
|-----|-------|------------------|
| 8 | SVMs | Kernel trick, when/why they work |
| 9 | kNN & Naive Bayes | Quick coverage, interview-style |
| 10 | **Gradient Boosting** | XGBoost, LightGBM, CatBoost |
| 11 | Hyperparameter Tuning | GridSearchCV, RandomizedSearch, Optuna |
| 12 | Model Evaluation | ROC, AUC, Precision-Recall, F1, Cross-validation |
| 13 | Feature Engineering | Scaling, encoding, PCA, feature selection |
| 14 | **Mini-Project**: Algorithm Comparison | Compare ML algorithms on one dataset |

### Week 3: Deep Learning Essentials
**Goal**: Modern DL without old-school derivations

| Day | Topic | Key Technologies |
|-----|-------|------------------|
| 15 | Neural Network Basics | Forward/backprop intuition (not full math) |
| 16 | **Building NN with PyTorch** | Tensors, autograd, nn.Module, training loops, BatchNorm |
| 17 | CNNs | Image classification (CIFAR/MNIST) |
| 18 | RNNs & LSTMs | Sequence modeling (text/sensor data) |
| 19 | Transformers | Attention mechanism, HuggingFace intro |
| 20 | Transfer Learning | Fine-tuning ResNet, BERT |
| 21 | **Mini-Project**: Sentiment Analysis | Image classification with transfer learning |

### Week 4: Advanced & Real-World ML
**Goal**: Industry-ready skills

| Day | Topic | Key Technologies |
|-----|-------|------------------|
| 22 | MLOps Basics | Model versioning, experiment tracking (MLflow) |
| 23 | Model Deployment | Flask/FastAPI + Docker + GitHub Actions |
| 24 | Data Pipelines | ETL basics, Airflow/Prefect overview |
| 25 | Unsupervised Learning | KMeans, DBSCAN, Autoencoders |
| 26 | Recommendation Systems | Collaborative filtering + embeddings |
| 27 | Time Series | Prophet, LSTMs for forecasting |
| 28 | **Mini-Project**: Deploy ML Model | Streamlit + HuggingFace spaces |

### Final 2 Days: Portfolio & Revision
| Day | Topic | Focus |
|-----|-------|-------|
| 29 | Concept Revision | Notes + cheat sheets |
| 30 | Portfolio Polish | GitHub repos, READMEs, results, visuals |

## Tech Stack

### Core ML Libraries
- **Scikit-learn**: Classical ML algorithms
- **XGBoost/LightGBM/CatBoost**: Gradient boosting
- **PyTorch**: Deep learning framework
- **HuggingFace**: Transformers and NLP

### Data & Visualization
- **Pandas/NumPy**: Data manipulation
- **Matplotlib/Seaborn**: Visualization
- **Plotly**: Interactive plots

### MLOps & Deployment
- **MLflow**: Experiment tracking
- **Docker**: Containerization
- **FastAPI/Flask**: API development
- **Streamlit**: Quick prototyping
- **GitHub Actions**: CI/CD

## Getting Started

### Prerequisites
- Python 3.8+
- Basic programming knowledge
- Familiarity with data structures

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/CodeByHashir/30-Days-Of-ML.git
cd 30-Days-Of-ML
```

2. **Create virtual environment**
```bash
python -m venv ml_env
source ml_env/bin/activate  # On Windows: ml_env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Start with Day 1**
```bash
jupyter notebook Day01_Python_DS_Refresher.ipynb
```

## Project Structure

```
30-days-of-ml/
├── 30-Days-Of-ML/                    # Main course directory
│   ├── Day01_Python_DS_Refresher.ipynb
│   ├── Day02_Data_Preprocessing.ipynb
│   ├── Day03_EDA_Best_Practices.ipynb
│   ├── Day04_ML_Pipeline_Intro.ipynb
│   ├── Day05_Linear_Logistic_Regression.ipynb
│   ├── Day06_Decision_Trees_Random_Forests.ipynb
│   ├── Day07_Titanic.ipynb
│   ├── Day08_SVMs_Kernel_Trick.ipynb
│   ├── Day09_kNN_Naive_Bayes.ipynb
│   ├── Day10_Gradient_Boosting.ipynb
│   ├── Day11_Hyperparameter_Tuning.ipynb
│   ├── Day12_Model_Evaluation.ipynb
│   ├── Day13_Feature_Engineering.ipynb
│   ├── Day14_Algorithm_Comparison.ipynb
│   ├── Day15_Neural_Network_Basics.ipynb
│   ├── Day16_Building_Neural_Networks_with_PyTorch.ipynb
│   ├── Day17_CNNs_Image_Classification.ipynb
│   ├── LICENSE
│   └── README.md
```

### Directory Breakdown

- **`30-Days-Of-ML/`**: Contains all the Jupyter notebook lessons organized by day
  - **Days 1-14**: Till - Deep learning fundamentals with PyTorch
  - **LICENSE**: MIT license file
  - **README.md**: Main course documentation

- **Root Level Files**:
  - **`30_days_of_ml_guide.md`**: Comprehensive course roadmap and learning objectives

### Current Progress Status

✅ **Completed Days (1-17)**:
- Day 1: Python for Data Science Refresher
- Day 2: Data Preprocessing
- Day 3: EDA Best Practices
- Day 4: ML Pipeline Introduction
- Day 5: Linear & Logistic Regression
- Day 6: Decision Trees & Random Forests
- Day 7: Titanic Dataset Project
- Day 8: SVMs & Kernel Trick
- Day 9: kNN & Naive Bayes
- Day 10: Gradient Boosting (XGBoost, LightGBM, CatBoost)
- Day 11: Hyperparameter Tuning
- Day 12: Model Evaluation
- Day 13: Feature Engineering
- Day 14: **Algorithm Comparison Mini-Project** - Comprehensive ML model comparison
- Day 15: Neural Network Basics
- Day 16: **Building Neural Networks with PyTorch** - Complete PyTorch tutorial with BatchNorm comparison, gradient analysis, and model checkpointing
 - Day 17: **Convolutional Neural Networks (CNNs)** - Image classification with PyTorch (MNIST + CIFAR-10), data augmentation, BatchNorm/Dropout, evaluation and visualizations

🔄 **In Progress**: Days 18-30 (to be added)

## Learning Objectives

By the end of this course, you will:

- **Master modern ML workflows** from data preprocessing to deployment
- **Build production-ready models** using industry best practices
- **Deploy ML models** with Docker, FastAPI, and cloud platforms
- **Implement MLOps pipelines** with experiment tracking and versioning
- **Work with real-world datasets** and solve actual business problems
- **Create a professional portfolio** showcasing your ML projects

## Projects Portfolio

Each week includes hands-on projects that you can add to your portfolio:

1. **Titanic Survival Prediction System** - End-to-end ML pipeline
2. **Algorithm Comparison Study** - Comprehensive model evaluation with 9 ML algorithms, cross-validation, and performance visualization
3. **Sentiment Analysis App** - Deep learning with transfer learning
4. **Deployed ML Service** - Production-ready API with monitoring

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Kaggle for providing excellent datasets
- HuggingFace for transformer models and tools
- The open-source ML community for continuous innovation

## Support

If you have any questions or need help with the course:

- Open an issue on GitHub
- Check the discussions section
- Join our community (coming soon)

---

**Ready to become an ML engineer in 30 days? Let's start this journey!**

*"The best way to learn Machine Learning is by building Machine Learning."*
