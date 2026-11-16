# Mobile Price Range Prediction Using Machine Learning

## Project Description

Mobile Price Range Prediction is a machine learning classification project that predicts the price category of a mobile phone based on its hardware specifications.

The project uses a Support Vector Machine (SVM) classifier as its main prediction model. It also compares the accuracy of SVM with other machine learning algorithms.

## Objectives

- Predict the price range of mobile phones.
- Study the relationship between mobile specifications and price categories.
- Train a classification model using SVM.
- Evaluate model performance using different evaluation metrics.
- Compare SVM with other machine learning algorithms.
- Develop a web application using Streamlit for user-friendly predictions.

## Dataset Features

The dataset contains the following input features:

1. Rear Main Camera (MP)
2. Rear Camera Count
3. Front Camera (MP)
4. RAM (GB)
5. Storage (GB)
6. Battery Capacity (mAh)
7. Display Size (inches)

**Target Variable:** `Price_Range_INR`

## Algorithms Used

- Support Vector Machine (SVM) – Main Model
- Random Forest
- Extra Trees Classifier
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Gradient Boosting

## Technologies and Libraries

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit

## Model Evaluation Metrics

The project evaluates the model using:

- Accuracy
- Precision
- Sensitivity (Recall)
- F1 Score
- Confusion Matrix
- Cohen's Kappa
- Matthews Correlation Coefficient (MCC)

## Web Application

The project includes a Streamlit web application that allows users to:

- View dataset information.
- Explore price range distribution.
- Check SVM model performance.
- View the confusion matrix.
- Compare machine learning model accuracy.
- Enter mobile specifications and predict the price category.

## Repository Structure

```text
Mobile-Price-Prediction/
│
├── app.py
├── mobile_price_dataset.csv
├── requirements.txt
└── README.md
```

## How to Run the Project Locally

1. Download or clone this repository.
2. Install the required libraries:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit application:

```bash
streamlit run app.py
```

4. Open the local URL displayed in your terminal.

## Deployment

This application can be deployed using Streamlit Community Cloud by connecting the GitHub repository and selecting `app.py` as the main file.

## Conclusion

This project demonstrates the use of machine learning classification techniques to predict mobile phone price categories based on hardware specifications. The Streamlit application provides an interactive interface for exploring model performance and making predictions.

**Project Type:** Machine Learning – Classification  
**Main Algorithm:** Support Vector Machine (SVM)  
**Application Framework:** Streamlit
