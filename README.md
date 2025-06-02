# Data Science Team 1
### 2025 Spring 



# The Impact of Redevelopment on Older Adults
Case Study of Seongnam City with Policy Recommendations

## 🏙️ Project Overview

This project analyzes the level of age-friendliness across cities and counties in Gyeonggi-do, with a focus on Seongnam-si, where urban redevelopment and population aging are occurring simultaneously. As Korea entered an aged society in 2017, older adults have become increasingly vulnerable, especially in redevelopment areas.

## 🔍 Key Objectives
	•	Evaluate the age-friendliness of Seongnam-si using provincial data.
	•	Identify key factors affecting elderly quality of life (e.g., transportation, welfare services, safety).
	•	Use machine learning to determine the main predictors of residential satisfaction.
	•	Propose policy recommendations to support residential stability and well-being for older adults during redevelopment.

Seongnam-si, especially its older districts (e.g., Sujeong-gu and Jungwon-gu), illustrates the urgent need to improve age-friendliness in the face of redevelopment. This project aims to guide policy toward more inclusive, senior-friendly urban planning.

## 📁 Directory Structure
each-city/
│
├── Data/
│   ├── 성남시 연도별 인구지표/
│   │   └── xlsx파일/
│   │       ├── 17년_연령별_인구.csv
│   │       ├── ...
│   │       └── 24년_연령별-인구.csv
│   ├── redevelopment_before_2017_2019.csv
│   └── redevelopment_after_2023.csv
│
├── Modeling/
│   ├── Classification/
│   │   ├── Classification.py
│   │   ├── log.txt
│   │   └── seongnam_classification_results/
│   │
│   ├── Ensemble/
│   │   ├── Ensemble.py
│   │   ├── log.txt
│   │   └── seongnam_ensemble_results/
│   │
│   └── Regression/
│       ├── Regression.py
│       ├── log.txt
│       └── seongnam_regression_results/
│
└── Preprocessing/
    └── preprocessing.ipynb


## ▶️ How to Run
	1.	Clone the repository and install dependencies

git clone https://github.com/hwm31/Seongnam-redevelopment-elderly-study.git
cd Seongnam-redevelopment-elderly-study
pip install -r requirements.txt


	2.	Data Preprocessing
	•	Run the Jupyter notebook:

jupyter notebook Preprocessing/preprocessing.ipynb


	•	This notebook loads the raw data under Data/, processes it, and prepares it for modeling.

	3.	Run Models
	•	You can execute each model by running the respective Python files:

python Modeling/Classification/Classification.py
python Modeling/Ensemble/Ensemble.py
python Modeling/Regression/Regression.py


	•	Output logs are saved in each folder’s log.txt, and result files are stored in the respective *_results/ directories.


## 💡 Result
