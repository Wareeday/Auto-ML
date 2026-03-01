## A Python-Based Automated Machine Learning Platform

 ## Overview

The **AutoML Web Application** is a lightweight yet scalable machine learning platform that enables users to upload datasets and automatically train, evaluate, and select the best-performing model.

The system performs:

* Automated data preprocessing
* Model training and evaluation
* Performance comparison
* Best model selection
* Interactive web-based results display

This project demonstrates the core principles of **Automated Machine Learning (AutoML)** using **Python and Flask**.

## Features

*  Upload CSV datasets through a web interface
*  Automatic categorical encoding
*  Train-test split automation
*  Multiple ML model comparison
*  Best model selection based on accuracy
*  Real-time result display
*  Modular and scalable architecture

##  Models Implemented

* Logistic Regression
* Random Forest Classifier
* Support Vector Machine (SVM)

*The architecture is designed to easily integrate additional models.*


##  Tech Stack

**Backend:** Python
**Framework:** Flask
**Machine Learning Libraries:** scikit-learn
**Data Processing:** Pandas, NumPy
**Frontend:** HTML (Jinja Templates)

##  How It Works

1. User uploads a CSV dataset.
2. User specifies the target column.
3. The system:

   * Encodes categorical variables
   * Splits data into training and testing sets
   * Trains multiple machine learning models
   * Evaluates performance
   * Selects the best-performing model
4. Displays the best model and its accuracy score.

##  Future Enhancements

* Hyperparameter tuning
* Cross-validation
* Regression task detection
* Model download/export functionality
* Interactive visualization dashboard
* Cloud deployment (AWS / GCP)
* Integration with Generative AI agents

##  Learning Outcomes

* Practical implementation of AutoML principles
* Model comparison and evaluation techniques
* Web-based machine learning deployment
* Automated data preprocessing pipelines
* Scalable AI system design
