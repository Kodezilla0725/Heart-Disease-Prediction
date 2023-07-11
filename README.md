# Heart-Disease-Prediction

## 1. Introduction
In modern healthcare, non-invasive diagnostic tests have become a crucial requirement, but the significant volume of electronic data poses challenges for accurate diagnosis of diseases. Supervised machine learning algorithms have been proven to outperform traditional diagnostic systems, aiding in the early detection of high-risk diseases. This project aims to develop a cost effective diagnostic system that utilizes fundamental health parameters to predict potential future health complications using machine learning models. The proposed system can be used as a reliable tool for early detection and prevention of chronic diseases, particularly in resource-limited settings. The integration of machine learning algorithms in healthcare has the potential to revolutionize the medical industry, enhance patient outcomes, and reduce healthcare costs.

This project utilises ML in Python to accurately predict Heart Disease using patient data, aiding in early disease detection and identifying high-risk individuals. It is invaluable in resource-limited settings, improving health outcomes, reducing costs, and enhancing quality of life for individuals and communities.


## 2. Heart Disease Model Implementation

### 2.1 Importing required Dataset and Libraries

Firstly, NumPy and Pandas are popular Python libraries used for numerical calculations and data manipulation respectively. scikit-learn is a machine learning library in Python that provides various tools for modeling data.

The `heart_data` variable represents a dataset of heart disease patients, which is stored in a CSV file. The data is loaded into a Pandas DataFrame using the `read_csv()` function.


### 2.2 Splitting Data into Training & Test Data and Model Training

The `X` variable represents the input features, which are all the columns of the DataFrame except the target column, which represents the target variable `Y` (the presence of heart disease in the patient). `train_test_split()` is a function from scikit-learn used to split the data into training and testing datasets. The function splits the input features and target variables into four separate variables - `X_train`, `X_test`, `Y_train`, and `Y_test`. The `test_size` parameter specifies the proportion of the data to be used for testing, while the `stratify` parameter ensures that the same proportion of patients with and without heart disease is present in both the training and testing sets. The `random_state` parameter ensures reproducibility of the split, so that the same results can be obtained across different runs of the code.

The `LogisticRegression()` function from `scikit-learn` creates a logistic regression model. Logistic regression is a classification algorithm that predicts the probability of the target variable belonging to a particular class. In this case, the model predicts the probability of a patient having heart disease based on their input features. The `fit()` method is used to train the model on the training set.


### 2.3 Testing

The `accuracy_score()` function from scikit-learn is used to measure the accuracy of the model on both the training and testing sets. The function compares the predicted target variable with the actual target variable and returns the fraction of correctly classified samples. This gives an idea of how well the model is performing.


### 2.4 Result Analysis

The `input_data` variable is a tuple containing the input features for a single patient. These features are used to make a prediction for that patient. The `np.asarray()` function converts the tuple to a NumPy array, which is then reshaped into a 2D array using `reshape()` so that the `predict()` function can handle it. The `predict()` function takes the input data and returns the predicted target variable.

Finally, the predicted target variable is checked to see if it corresponds to a patient with or without heart disease. If the predicted value is 0, then the patient is predicted to not have heart disease. If the predicted value is 1, then the patient is predicted to have heart disease. This is outputted as a message to the user.


## 3. Conclusion and Future Scope

### 3.1 Conclusion

The use of different ML algorithms enabled the early detection of many chronic diseases such as heart, kidney, breast cancer, and brain diseases, diabetes, Parkinson’s, etc. Throughout the literature, Support Vector Machine (SVM) and Logistic Regression algorithms were the most widely used at prediction, while accuracy was the most used performance metric. Both models proved to be adequate at predicting common diseases.

These algorithms utilize a variety of methods, including feature selection, data pre-processing, and model training, to find patterns and correlations in the datasets and forecast the risk of a disease occurring accurately. Early disease diagnosis can lead to proactive treatments and better patient outcomes. To guarantee these algorithms' efficiency and dependability, it is crucial to validate them using extensive clinical research. There is a lot of potential for these algorithms to be incorporated into clinical practice and enhance patient care with ongoing improvements in machine learning techniques and the accessibility of huge datasets.

### 3.2 Future Scope

In future research, there is a need to develop more sophisticated machine learning algorithms to improve the accuracy of disease prediction. It is also important to regularly calibrate the learning models after the training phase for better performance. Moreover, expanding datasets to include diverse demographics is crucial to avoid overfitting and increase the precision of the deployed models. Finally, using more relevant feature selection techniques can further enhance the performance of the learning models.

This Multiple Disease Prediction System can be integrated into electronic health records to provide real-time predictions for patients, enabling doctors to make prompt and informed decisions. Additionally, it can be utilized for public health monitoring and disease surveillance, facilitating the identification and rapid response to outbreaks. As machine learning advances, we can expect more sophisticated and advanced disease prediction systems to emerge, transforming the way we diagnose and treat diseases.


## 4. References

[Heart Disease Dataset](https://www.kaggle.com/datasets/aavigan/cleveland-clinic-heart-disease-dataset)
