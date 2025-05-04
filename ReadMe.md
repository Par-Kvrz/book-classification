# Text Classification Project - University of Ottawa (Data Science Applications Course)


This project is a text classification pipeline that processes books from [Project Gutenberg](https://www.gutenberg.org/) and uses machine learning to predict their genres. It was originally developed in a Jupyter Notebook and later exported as a `.py` script, so you'll see `In[]` cell markers in the code.

## 📅 Timeline

This project was completed in February 2020 as part of my Master's degree coursework and uploaded to GitLab in 2025 for portfolio purposes.


## 📚 Overview

The goal of the project is to classify literary texts based on their genre using classic NLP techniques and machine learning models.

## 📂 Data

The dataset consists of **7 books** downloaded from Project Gutenberg. Each book is assumed to represent a distinct genre or authorial style.

## 🧹 Preprocessing

The script uses **NLTK** for natural language preprocessing:

- Lowercasing all text
- Tokenization
- Stop word removal
- Creating random text samples from each book
- Assigning genre labels to each sample

## ⚙️ Feature Engineering

Two techniques are used to convert text into numerical feature vectors:

- **Bag of Words (BoW)**
- **TF-IDF (Term Frequency–Inverse Document Frequency)**

## 🧠 Models

Three supervised learning models are trained and compared:

- Support Vector Machine (**SVM**)
- K-Nearest Neighbors (**KNN**)
- Decision Tree Classifier

## 📈 Evaluation & Analysis

The project includes:

- Accuracy comparison across models
- Error analysis of misclassified samples
- Word cloud visualization of commonly misclassified terms
- Prediction score distributions

