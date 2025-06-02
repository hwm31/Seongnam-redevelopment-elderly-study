# 🏘️ The Impact of Redevelopment on Older Adults
### Case Study of Seongnam City with Policy Recommendations
*Data Science Team 1 | 2025 Spring*

<div align="center">

![Project Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

</div>

---

## 🌟 Project Overview

This comprehensive data science project analyzes the **age-friendliness** across cities and counties in **Gyeonggi-do Province**, with a special focus on **Seongnam-si**, where urban redevelopment and population aging intersect. As Korea transitioned into an aged society in 2017, older adults have become increasingly vulnerable, particularly in redevelopment zones.

> 📊 **Key Insight**: Seongnam-si's older districts (Sujeong-gu and Jungwon-gu) exemplify the urgent need for age-friendly urban planning during redevelopment processes.

---

## 🎯 Research Objectives

<table>
<tr>
<td width="50%">

### 📈 **Primary Goals**
- **Evaluate** age-friendliness metrics in Seongnam-si
- **Identify** key factors affecting elderly quality of life
- **Apply** machine learning for residential satisfaction prediction
- **Propose** evidence-based policy recommendations

</td>
<td width="50%">

### 🔍 **Focus Areas**
- 🚌 Transportation accessibility
- 🏥 Healthcare & welfare services  
- 🛡️ Safety & security measures
- 🏠 Housing stability during redevelopment

</td>
</tr>
</table>

---

## 📂 Project Structure

```
📁 each-city/
│
├── 📊 Data/
│   ├── 🗂️ 성남시 연도별 인구지표/
│   │   └── 📋 xlsx파일/
│   │       ├── 17년_연령별_인구.csv
│   │       ├── 18년_연령별_인구.csv
│   │       ├── ...
│   │       └── 24년_연령별-인구.csv
│   ├── 🏗️ redevelopment_before_2017_2019.csv
│   └── 🆕 redevelopment_after_2023.csv
│
├── 🤖 Modeling/
│   ├── 🎯 Classification/
│   │   ├── Classification.py
│   │   ├── log.txt
│   │   └── 📈 seongnam_classification_results/
│   │
│   ├── 🔄 Ensemble/
│   │   ├── Ensemble.py
│   │   ├── log.txt
│   │   └── 📊 seongnam_ensemble_results/
│   │
│   └── 📉 Regression/
│       ├── Regression.py
│       ├── log.txt
│       └── 📋 seongnam_regression_results/
│
└── 🔧 Preprocessing/
    └── preprocessing.ipynb
```

---

## 🚀 Quick Start Guide

### 1️⃣ **Setup & Installation**
```bash
# Clone the repository
git clone https://github.com/hwm31/Seongnam-redevelopment-elderly-study.git

# Navigate to project directory
cd Seongnam-redevelopment-elderly-study

# Install required dependencies
pip install -r requirements.txt
```

### 2️⃣ **Data Preprocessing**
```bash
# Launch Jupyter Notebook
jupyter notebook Preprocessing/preprocessing.ipynb
```
> 💡 This notebook loads raw data from `Data/`, processes it, and prepares datasets for machine learning models.

### 3️⃣ **Run Machine Learning Models**
Execute each model independently:

```bash
# Classification Model
python Modeling/Classification/Classification.py

# Ensemble Methods
python Modeling/Ensemble/Ensemble.py

# Regression Analysis
python Modeling/Regression/Regression.py
```

> 📝 **Output**: Logs are saved in `log.txt` files, and detailed results are stored in respective `*_results/` directories.

---

## 📊 Key Features

<div align="center">

| Feature | Description | Status |
|---------|-------------|--------|
| 🗂️ **Multi-year Analysis** | Population data from 2017-2024 | ✅ Complete |
| 🤖 **ML Pipeline** | Classification, Regression & Ensemble | ✅ Complete |
| 📈 **Policy Insights** | Data-driven recommendations | 🔄 In Progress |
| 🎯 **Age-Friendly Metrics** | Comprehensive evaluation framework | ✅ Complete |

</div>

---

## 💡 Expected Outcomes

This research aims to provide:

- 📋 **Comprehensive assessment** of Seongnam-si's age-friendliness
- 🎯 **Predictive models** for residential satisfaction among older adults
- 📜 **Policy recommendations** for inclusive urban redevelopment
- 🌉 **Framework** applicable to other aging cities in Korea

---

## 🤝 Contributing

We welcome contributions! Please feel free to:
- 🐛 Report bugs or issues
- 💡 Suggest new features or improvements  
- 📝 Improve documentation
- 🔧 Submit pull requests

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Team

**Data Science Team 1** | 2025 Spring Semester

---

<div align="center">

**🏙️ Building Age-Friendly Cities Through Data Science 🏙️**

*Making urban redevelopment inclusive for all generations*

</div>
