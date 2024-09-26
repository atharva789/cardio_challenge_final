# Heartfelt Predictions: Machine Learning to Avoid Cardiac Disasters

## Overview

This repository contains machine learning models developed as part of **Team 9**'s work for the **Cardio Challenge** under the auspices of the U.S. Department of Energy at **Lawrence Livermore National Laboratory (LLNL)**. The primary focus is on detecting cardiac abnormalities using ECG data and predicting activation times in the myocardium. The repository includes various binary and multi-class classification models as well as advanced techniques for feature extraction and activation map reconstruction.

## Team
- **Atharva Gupta**
- Anuvetha Govindarajan
- Brady Snyder
- Davinia Muthalaly
- Noah Gallego

## Solutions

### 1. **Binary Heartbeat Classification**
   - **Problem Solved**: Identifying normal versus abnormal heartbeats based on 10-lead ECG data.
   - **Solution**: Implemented a **logistic regression model** with 5-fold cross-validation, achieving an accuracy of **83%** on test data. This solution provides a baseline model for quick diagnosis of irregular heartbeats.

### 2. **Neural Network for Binary Classification**
   - **Problem Solved**: Enhancing the detection of normal and abnormal heartbeats.
   - **Solution**: A **feed-forward neural network** with ReLU activation functions in hidden layers and sigmoid activation in the output layer was implemented. This model improved accuracy over logistic regression.

### 3. **Multi-Class Arrhythmia Detection**
   - **Problem Solved**: Classifying ECG sequences into one of five arrhythmia categories.
   - **Solution**: Built multiple models, including:
     - **K-Nearest Neighbors (KNN)**: Achieved an F1 score of **0.97** for normal cases but lower for other classes due to data imbalance.
     - **Decision Trees**: With a max depth of 10, produced an average F1 score of **0.72** for the majority class.
     - **Convolutional Neural Networks (CNN)**: This model achieved **99%** accuracy across all classes, solving the imbalance problem through data rebalancing techniques (down-sampling and up-sampling).

### 4. **Activation Map Reconstruction and Prediction**
   - **Problem Solved**: Predicting the activation time across 75 distinct regions of the myocardium.
   - **Solution**: A **hybrid network architecture** was developed, combining convolutional and pooling layers with fully connected layers. The model achieved a **mean-squared error (MSE) of 104**, with predictions within **5.93 milliseconds** of actual activation times.

### 5. **Fourier Transform for Signal De-noising**
   - **Problem Solved**: Reducing noise in ECG signal data for improved feature extraction and model accuracy.
   - **Solution**: Applied a **Fourier Transform** to activation time sequences, resulting in a de-noised dataset that improved CNN performance, bringing the MSE down to **104** from previous models.

### 6. **Grad-CAM for CNN Interpretability**
   - **Problem Solved**: Lack of interpretability in neural networks, particularly CNNs.
   - **Solution**: Integrated **Grad-CAM** to visualize which features influence the CNN's decision-making process, enhancing the model's transparency and aiding medical professionals in understanding the basis for predictions.

## Key Metrics
- **Data Imbalance**: Addressed through rebalancing techniques.
- **Accuracy**: Best model (CNN) achieved **99%** accuracy.
- **Activation Time Prediction**: Achieved predictions within **5.93ms** of actual times.
