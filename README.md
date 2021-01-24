# Vehicle-Insurance-Claim-Amount-Prediction-MLR-with-GradientBoostingRegressor
Python Data Science Project, Vehicle-Insurance-Claim-Amount-Prediction using MLR algo with GradientBoostingRegressor:

- 79.62% accuracy (MLR)
- 89.92% accuracy (GradientBoostingRegressor)
- Multiple Regression Model Overall F-test for Regression Line Slope = MSR/MSE = 571.7
- RMSE: 10087 (MLR)
- Kurtosis/Peakedness=[M4/M2^2]=3.539
- Skewness=Mean-Mode=Negative=-0.061
- RMSE: 6597 (GradientBoostingRegressor)
- MAE: 8238 (MLR)
- MAE: 5186 (GradientBoostingRegressor)
- Mallow's_CP: 0.001 (MLR & GradientBoostingRegressor)
- D_Watson: 1.919
- No Multicollinearity
- Homoscedastic Residual/Error Distribution

## Problem Statement
An Indian vehicle insurance company wants to create such a system through which the company can estimate proper insurance amount for Client's vehicle. If the company gets proper estimates for the insurance amount it can build better insurance plans & policies benefiting both sides.

They have contracted an consultency firm to understand the factors on which the insurance amount of cars depends. Specifically, they want to understand the factors affecting the insurance of cars in the Indian market. The company exactly wants to know:

- Which variables are significant in predicting the insurance amount.

- How well those variables describe the Insurance amount.

Based on various market surveys, the consulting firm has gathered a large dataset of different types of cars Features & Insurance details.

## About the Car Insurance Prediction
Insurance Claim Amount is Target variable & 4 features are Vehicle Length, CC, Weight, Age

## Business Goal
You are required to model the Insurance amount of cars with the available independent variables. It will be used by the management to understand how exactly the amount varies with the independent variables. They can accordingly manipulate the insurance schemes, conditions, terms & policies to meet the company goals. Further, the model will be a good way for management to understand the vehicle insurance dynamics of Indian market.

## Project Overview
In this Project I have unleashed the useful Data Science insights using this Vehicle Insurance dataset and performed the feature selection precisely to build multiple linear regression model along with GradientBoostingRegressor by combining the power of best statistical rules & principles to maximise accuracy at its best. The best thing is my model is not having any Multicollinearity & Heteroscedasticity problem.

## This Project is divided into 24 major steps which are as follows:
1. [Check out the Data](#data-check)
2. [Importing Libraries & setting up environment](#imp-lib)
3. [Loading dataset](#data-load)
4. [Data Cleaning & Preprocessing](#prep-clean)
5. [Shapiro for Normality test of All Numeric Columns](#shapiro-norm)
6. [Spearman Correlation Test for Measures of Association between non-normal Datas](#spear-corr)
7. [Outlier Treatment/Check](#out-check)
8. [Skewness Check](#skew-check)
9. [Exploratory Data Analysis ( EDA )](#data-expo)
10. [Assigning Label & Features](#Labe-Feature)
11. [Scaling of Numeric Features](#scale-feature)
12. [Train & Test Split](#data-split)
13. [Features P-Value & VIF Check](#p-vif)
14. [Final Implimentation of the MLR Model](#final-model)
15. [Model Evaluation](#mod-eval)
16. [Actual vs Predicted Price of Used CAR](#actual-predicted)
17. [Residual Distribution of Predicted Used CAR Price](#re-dit)
18. [Amount of Error in Used CAR Price Prediction](#amt-er)
19. [Durbin Watson Auto-Correlation Test](#dur-wat)
20. [Regression Evaluation Metrics](#mod-eval)
21. [Plotting the Regression Line](#reg-plot)
22. [Heteroscedasticity Tests](#het-test)
23. [Auto-Correlation plot](#auto-plot)
24. [Gradient Boosting For Regression](#grad-boost)

## Selected Features for the Final Model
**1. Vehage:** Vehicle Age

**2. CC:** Vehicle CC

**3. Length:** Vehicle Length

