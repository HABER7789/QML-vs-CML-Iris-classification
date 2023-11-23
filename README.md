# Quantum vs Classical Machine Learning on the Iris Dataset

## Overview

This project compares the performance of a Quantum Machine Learning (QML) model with a Classical Machine Learning (CML) model on the Iris dataset. The QML model uses a basic quantum circuit implemented with Qiskit, while the CML model employs logistic regression from scikit-learn.

## Dataset

The Iris dataset is a well-known dataset in machine learning, consisting of measurements of four features of iris flowers: sepal length, sepal width, petal length, and petal width. The goal is to classify iris flowers into one of three species based on these features.

## Methodology

1. **Data Preprocessing:**
   - The Iris dataset features (sepal length, sepal width, petal length, and petal width) are standardized using scikit-learn's `StandardScaler`.
   - Labels, representing three different iris species, are initially mapped to binary values (0 or 1) for binary classification.

2. **Quantum Circuit:**
   - A simple quantum circuit is designed with Qiskit, consisting of two rotational gates (`rx` and `ry`) and a measurement operation.
   - The circuit is transpiled and run on the Aer simulator.

3. **Optimization:**
   - The quantum circuit parameters are optimized using classical optimization methods provided by `scipy.optimize.minimize`.

4. **Quantum Predictions:**
   - The optimized quantum circuit is used to make predictions on the test set.

5. **Classical Logistic Regression:**
   - A classical logistic regression model is trained using scikit-learn.

6. **Classical Predictions:**
   - The trained classical model is used to make predictions on the test set.

7. **Accuracy Comparison:**
   - The accuracy of both the quantum and classical models is calculated using scikit-learn's `accuracy_score`.

## Results

- **Quantum ML Accuracy:** 66.67%
- **Classical ML Accuracy:** 100.00%

## Observations

- The quantum machine learning model achieved lower accuracy compared to the classical logistic regression model on the Iris dataset.

## Conclusion

So, from my observation and implementation, the project clearly states that Quantum ML accuracy is nowhere near Classical ML accuracy. However, classical machine learning has evolved over years and has had extensive research in comparison to Quantum Machine learning.

The implementation I did was by using a mere Simple Quantum Circuit, whereas the classical Model which is using logistic regression has much more complexity in comparison to the current quantum circuit. The sole reason would be, I have insufficient knowledge to develop a more complex quantum circuit using different parameters and gate combinations. As this is just the beginning of my journey in quantum computing, I would consider this as an experiment done for learning purposes.

Not all tasks done by classical ML would have better accuracy in Quantum ML, I believe as I move further and study more about how a complex quantum circuit with less noise can be implemented, I will attain the sufficient intellectual for gaining the accuracy of 100% using QML. Also QML is used for far more complex computations in comparison to the classification that I did in the project, so quantum computers do work better at such tasks than the classical computer.

# Dhairya Patel
