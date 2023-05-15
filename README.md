# Churn Rate Telco Kaggle

This repository contains code and datasets related to predicting churn rate for Telco Customers. The dataset was obtained from Kaggle, and you can find it [here](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

Created by Albert Lleida, May 2023.

## Table of Contents
- [Introduction](#introduction)
- [Overview](#overview)
- [Kernel 1: Predicting Churn Rate - Logistic Regression, Decision Tree, and Random Forest Classifier and Logistic Regression (SMOTE).ipynb](#kernel-1-predicting-churn-rate---logistic-regression-decision-tree-and-random-forest-classifier-smoteipynb)
- [Kernel 2: Predicting Churn Rate - Customer Profiles & Exploratory Data Analysis.ipynb](#kernel-2-predicting-churn-rate---customer-profiles--exploratory-data-analysisipynb)
- [Tableau Data Exploration](#tableau-data-exploration)
- [Files](#files)
- [Technologies Used](#technologies-used)

## Introduction
This repository contains code and datasets related to predicting churn rate for Telco Customers. The goal of this project is to predict churn rate using various machine learning models. The dataset was obtained from Kaggle and contains information about Telco customers such as their demographics, services availed, and churn status. The project involves data cleaning and wrangling, data distribution analysis, and model evaluation using different classifiers.

## Overview
The project consists of two Jupyter Notebooks that provide a comprehensive analysis of the Telco customer churn dataset.

## Kernel 1: Predicting Churn Rate - Logistic Regression, Decision Tree, and Random Forest Classifier-SMOTE.ipynb
In this kernel, the churn rate for Telco Customers is predicted using logistic regression, decision tree, and random forest classifiers. The notebook begins with data cleaning and wrangling, followed by an analysis of the data distribution. Logistic regression is applied first, and then the results are compared with other models such as decision tree and random forest classifiers. The performance of each model is evaluated using metric scores, and the results are summarized in the confusion matrix.

## Kernel 2: Predicting Churn Rate - Customer Profiles & Exploratory Data Analysis.ipynb
In this kernel, the main focus is on examining the main characteristics of customers who were detected and not detected by the logistic regression model. The logistic regression model is applied with oversampled data, as it yielded the highest number of true positives (predicted churn). The analysis aims to identify the features that the company should consider while improving the model to reduce the false negative rate. Additionally, exploratory data analysis is conducted on current customers to determine which profiles are more valuable or lucrative for the company.

## Tableau Data Exploration
In addition to the Jupyter Notebooks, Tableau is used for further data exploration and visualization. The Tableau dashboard provides interactive visualizations to gain deeper insights into the churn rate and customer behavior. The Tableau dashboard can be accessed [here](https://public.tableau.com/app/profile/albert1030/viz/ChurnRate_16839185900520/Historia1?publish=yes).

## Files
The "Files" folder contains the following datasets:
- Original dataset: The raw dataset obtained from Kaggle.
- Clean dataset: The dataset after undergoing data cleaning and wrangling (clean_data_churn.csv).
- Test dataset: This dataset includes a column with the predictions of Logistic Regression - SMOTE model (test_data_churn).

## Technologies Used
- Jupyter Notebook
- Python
- pandas
- numpy
- matplotlib
- seaborn
- Logistic Regression
- Decision Tree
- Random Forest Classifier
