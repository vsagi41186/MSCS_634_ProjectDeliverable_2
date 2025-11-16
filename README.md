# MSCS_634_ProjectDeliverable_2

# Breast Cancer Regression Analysis README

## Overview
This project performs regression analysis on a breast cancer dataset to predict the survival months of patients using two regression models: **Linear Regression** and **Ridge Regression**. The dataset was processed, and missing values were handled before building the models. Cross-validation and evaluation metrics like **R2**, **RMSE**, and **Cross-validation MSE** were used to compare the models' performance.

### Key Steps:
1. **Data Preprocessing**:
   - **Missing values**: Missing values in numeric columns are filled with the median, and missing values in categorical columns are filled with the mode.
   - **Feature Encoding**: Categorical columns are encoded using label encoding to transform them into numeric values suitable for regression models.

2. **Modeling**:
   - **Linear Regression**: A simple linear regression model was trained on the dataset.
   - **Ridge Regression**: A ridge regression model with L2 regularization was trained to prevent overfitting.

3. **Evaluation**:
   - **R2 Score**: Measures the proportion of variance in the target variable explained by the model.
   - **RMSE**: Root Mean Squared Error measures the average error between predicted and actual values.
   - **Cross-validation MSE**: Mean Squared Error is computed using 5-fold cross-validation to assess the generalizability of the models.

### Results:

#### Linear Regression Evaluation:
- **R2**: 0.060
- **RMSE**: 22.419
- **Cross-validation (MSE)**: -514.489

#### Ridge Regression Evaluation:
- **R2**: 0.060
- **RMSE**: 22.419
- **Cross-validation (MSE)**: -514.480

#### Best Model:
- Based on the **R2** score and other metrics, **Ridge Regression** was chosen as the best-performing model since it showed a slightly better cross-validation performance compared to **Linear Regression**.


## Challenges and Next Steps

During the regression analysis, several challenges were encountered, and addressing them can further improve the model's performance:

1. **Handling Imbalanced Data**: 
   - The dataset may have an imbalance between classes (e.g., Stage of cancer or survival status). This could impact model performance, especially in regression tasks. Future work could include techniques such as oversampling or undersampling to address class imbalances.

2. **Feature Engineering and Selection**:
   - Although basic encoding was applied, more sophisticated feature engineering could be used to improve model performance. For example, creating interaction features or using domain knowledge to generate additional meaningful features could potentially improve model accuracy.

3. **Hyperparameter Tuning**:
   - Both Linear and Ridge regression models used default hyperparameters. For better performance, hyperparameter tuning using techniques like GridSearchCV or RandomizedSearchCV can be applied to find the optimal settings for the models.

4. **Handling Multicollinearity**:
   - In the case of the Ridge regression, multicollinearity between predictor variables could affect model performance. Techniques such as Variance Inflation Factor (VIF) analysis could be employed to identify highly correlated predictors and remove them if necessary.

5. **Outliers**:
   - Outliers in numerical features (e.g., tumor size or survival months) may skew regression results. Future steps could include identifying and handling outliers through methods like Z-score or IQR to ensure model robustness.


## Requirements

- Python 3.x
- Libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`

Install the required libraries using the following:

```bash
pip install pandas numpy scikit-learn
```

## File Structure

- `Breast_Cancer.csv`: The dataset used in this analysis.
- `regression_analysis.py`: The Python script containing the full analysis and model evaluation code.

## Conclusion

This analysis demonstrates how to prepare data, perform regression modeling, and evaluate the models for predicting patient survival months based on various clinical attributes. While both models performed similarly, Ridge Regression was selected due to slightly better cross-validation results.

Challenges like feature engineering, hyperparameter tuning, and handling imbalanced data have been identified as future directions to further improve the model.



