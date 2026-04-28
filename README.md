# Teaching Documents to Speak  
NLP Feasibility for Legal Document Classification
An NLP pipeline that classifies how legal cases are treated using TF-IDF and machine learning in order to utilize in a regulatory environment
## Business Problem

Manual document review in legal and regulatory environments is slow, expensive, and error-prone. 
Understanding how prior cases are used is critical for legal strategy 
and compliance workflows.

This project explores whether NLP can automate this classification task at scale.
## Solution Overview

This project builds an NLP classification pipeline:

- Input: Raw legal text (`case_text`)
- Output: Treatment category (`case_outcome`)
- Classes: cited, referred to, applied, followed, considered

Pipeline:
1. Text preprocessing
2. TF-IDF vectorization
3. Machine learning classification (Logistic Regression, Random Forest)
4. Evaluation using accuracy, precision, recall, and F1-score
5. ## Dataset

- Source: Kaggle Legal Text Classification Dataset
- ~25,000 legal documents
- Key columns:
  - `case_text` → input text
  - `case_outcome` → target label

Due to class imbalance, the model focuses on the top 5 categories:
cited, referred to, applied, followed, considered.
## Model & Methods

- TF-IDF Vectorizer (max_features=5000, ngrams 1–2)
- Models:
  - DummyClassifier (baseline)
  - Logistic Regression (interpretable)
  - Random Forest (best performance)

Evaluation:
- Train/test split (80/20, stratified)
- Confusion matrices
- 10-fold cross-validation (Logistic Regression)
- Feature importance analysis
## Results

  Model                Accuracy  

  Naive Baseline       53.0%   
  Logistic Regression  57.3%    
  Random Forest        63.0%    

Key insight:
The Random Forest model improves performance by +10 percentage points over the baseline.

Per 1,000 documents:
- Baseline: ~530 correct
- Model: ~630 correct
- +100 additional correct classifications
