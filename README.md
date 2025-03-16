Project: Capstone Project 2 (Final)
===

# Author: Vijay Chaganti

*Data Source: [https://github.com/chagantvj/PracticalApplicationM17/blob/main/bank-additional-full.csv](https://github.com/chagantvj/CapstoneProject1/blob/main/US_Housing_Data.csv)*

*Python Code: https://github.com/chagantvj/CapstoneProject1/blob/main/VijayChaganti-CapStoneProjecct_Final.ipynb*

# Project overview and goals
The goal of this project is to identify more effective ways for predicting housing price based on given dataset. We will be training and tuning different set of classification & regression models to accurately predict the price. We will then evaluate and compare the models' performances to identify the best one, then further scrutinize it to find the most effective features that enhance performance.

Dataset information
---
*This dataset is very rich in number of columns that will help implementing several different encouding techniques for catorigical data*
*In combination with numerical and categorical data, its possible to generete good pipeline models to predict House Scale Price*

*The provided dataset about US Housing Scales

*Given dataset has 1460 entries with 80 columns*

<img width="1100" alt="Screenshot 2025-02-25 at 8 59 49 PM" src="https://github.com/user-attachments/assets/12b1baaa-91aa-4fc2-bf1c-759c9eba177d" />

**Date Understanding and Cleaning**

```
RangeIndex: 1460 entries, 0 to 1459
Data columns (total 81 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   Id             1460 non-null   int64  
 1   MSSubClass     1460 non-null   int64  
 2   MSZoning       1460 non-null   object 
 3   LotFrontage    1201 non-null   float64
 4   LotArea        1460 non-null   int64  
 5   Street         1460 non-null   object 
 6   Alley          91 non-null     object 
 7   LotShape       1460 non-null   object 
 8   LandContour    1460 non-null   object 
 9   Utilities      1460 non-null   object 
 10  LotConfig      1460 non-null   object 
 11  LandSlope      1460 non-null   object 
 12  Neighborhood   1460 non-null   object 
 13  Condition1     1460 non-null   object 
 14  Condition2     1460 non-null   object 
 15  BldgType       1460 non-null   object 
 16  HouseStyle     1460 non-null   object 
 17  OverallQual    1460 non-null   int64  
 18  OverallCond    1460 non-null   int64  
 19  YearBuilt      1460 non-null   int64  
 20  YearRemodAdd   1460 non-null   int64  
 21  RoofStyle      1460 non-null   object 
 22  RoofMatl       1460 non-null   object 
 23  Exterior1st    1460 non-null   object 
 24  Exterior2nd    1460 non-null   object 
 25  MasVnrType     588 non-null    object 
 26  MasVnrArea     1452 non-null   float64
 27  ExterQual      1460 non-null   object 
 28  ExterCond      1460 non-null   object 
 29  Foundation     1460 non-null   object 
 30  BsmtQual       1423 non-null   object 
 31  BsmtCond       1423 non-null   object 
 32  BsmtExposure   1422 non-null   object 
 33  BsmtFinType1   1423 non-null   object 
 34  BsmtFinSF1     1460 non-null   int64  
 35  BsmtFinType2   1422 non-null   object 
 36  BsmtFinSF2     1460 non-null   int64  
 37  BsmtUnfSF      1460 non-null   int64  
 38  TotalBsmtSF    1460 non-null   int64  
 39  Heating        1460 non-null   object 
 40  HeatingQC      1460 non-null   object 
 41  CentralAir     1460 non-null   object 
 42  Electrical     1459 non-null   object 
 43  1stFlrSF       1460 non-null   int64  
 44  2ndFlrSF       1460 non-null   int64  
 45  LowQualFinSF   1460 non-null   int64  
 46  GrLivArea      1460 non-null   int64  
 47  BsmtFullBath   1460 non-null   int64  
 48  BsmtHalfBath   1460 non-null   int64  
 49  FullBath       1460 non-null   int64  
 50  HalfBath       1460 non-null   int64  
 51  BedroomAbvGr   1460 non-null   int64  
 52  KitchenAbvGr   1460 non-null   int64  
 53  KitchenQual    1460 non-null   object 
 54  TotRmsAbvGrd   1460 non-null   int64  
 55  Functional     1460 non-null   object 
 56  Fireplaces     1460 non-null   int64  
 57  FireplaceQu    770 non-null    object 
 58  GarageType     1379 non-null   object 
 59  GarageYrBlt    1379 non-null   float64
 60  GarageFinish   1379 non-null   object 
 61  GarageCars     1460 non-null   int64  
 62  GarageArea     1460 non-null   int64  
 63  GarageQual     1379 non-null   object 
 64  GarageCond     1379 non-null   object 
 65  PavedDrive     1460 non-null   object 
 66  WoodDeckSF     1460 non-null   int64  
 67  OpenPorchSF    1460 non-null   int64  
 68  EnclosedPorch  1460 non-null   int64  
 69  3SsnPorch      1460 non-null   int64  
 70  ScreenPorch    1460 non-null   int64  
 71  PoolArea       1460 non-null   int64  
 72  PoolQC         7 non-null      object 
 73  Fence          281 non-null    object 
 74  MiscFeature    54 non-null     object 
 75  MiscVal        1460 non-null   int64  
 76  MoSold         1460 non-null   int64  
 77  YrSold         1460 non-null   int64  
 78  SaleType       1460 non-null   object 
 79  SaleCondition  1460 non-null   object 
 80  SalePrice      1460 non-null   int64  
dtypes: float64(3), int64(35), object(43)
```

```
numerical_columns = df.select_dtypes(['int64', 'float64']).columns
print(f"Total numerical features in dataset is: {len(numerical_columns)}")
   >>> Total numerical column in dataset is: 38

categorical_columns = df.select_dtypes(['object']).columns
print(f"Total categorical features in dataset is: {len(categorical_columns)}")
   >>> Total categorical features in dataset is: 43
```

<img width="1108" alt="Screenshot 2025-02-25 at 9 16 21 PM" src="https://github.com/user-attachments/assets/5042a981-4647-4069-bbb2-117bb6084598" />

```
emptyCount_PerFeature = df.isna().sum()
emptyCountPercent_PerFeature = (emptyCount_PerFeature / len(df)) * 100
nonZero_emptyCountPercent_PerFeature = emptyCountPercent_PerFeature[emptyCountPercent_PerFeature > 0]
rounded_Percent = np.round(nonZero_emptyCountPercent_PerFeature)
print(rounded_Percent)
```
<img width="1108" alt="Screenshot 2025-02-25 at 10 36 58 PM" src="https://github.com/user-attachments/assets/55c29b8d-18f7-4fb0-a20d-e762b7f41cf4" />


```
features_to_drop = emptyCountPercent_PerFeature[emptyCountPercent_PerFeature > 59].index
ddf = df.drop(columns=features_to_drop)
ddf.shape
    >>> (1460, 76)
```

*Removing Columns with Missing data and also some of the coulumns that are having many unique values that does play an important role in price*
---
```
From the data set above, there is lot of missing data for columns named Alley, MasVnrType, FireplaceQu, PoolQC, Fence, MiscFeature.
Hence, these columns are removed from data set to build model.

Unique values in column MSZoning: ['RL' 'RM' 'C (all)' 'FV' 'RH']
Unique values in column Neighborhood: ['CollgCr' 'Veenker' 'Crawfor' 'NoRidge' 'Mitchel' 'Somerst' 'NWAmes', 'OldTown' 'BrkSide'
                                       'Sawyer' 'NridgHt' 'NAmes' 'SawyerW' 'IDOTRR' 'MeadowV' 'Edwards' 'Timber' 'Gilbert' 'StoneBr'
                                        'ClearCr' 'NPkVill' 'Blmngtn' 'BrDale' 'SWISU' 'Blueste']

Unique values in column HouseStyle: ['2Story' '1Story' '1.5Fin' '1.5Unf' 'SFoyer' 'SLvl' '2.5Unf' '2.5Fin']
Unique values in column RoofStyle: ['Gable' 'Hip' 'Gambrel' 'Mansard' 'Flat' 'Shed']
Unique values in column RoofMatl: ['CompShg' 'WdShngl' 'Metal' 'WdShake' 'Membran' 'Tar&Grv' 'Roll' 'ClyTile']
Unique values in column Exterior2nd: ['VinylSd' 'MetalSd' 'Wd Shng' 'HdBoard' 'Plywood' 'Wd Sdng' 'CmentBd' 'BrkFace' 'Stucco' 'AsbShng' 'Brk Cmn' 'ImStucc' 'AsphShn' 'Stone' 'Other' 'CBlock']
Unique values in column Foundation: ['PConc' 'CBlock' 'BrkTil' 'Wood' 'Slab' 'Stone']
Unique values in column Heating: ['GasA' 'GasW' 'Grav' 'Wall' 'OthW' 'Floor']
Unique values in column Electrical: ['SBrkr' 'FuseF' 'FuseA' 'FuseP' 'Mix' nan]
Unique values in column SaleType: ['WD' 'New' 'COD' 'ConLD' 'ConLI' 'CWD' 'ConLw' 'Con' 'Oth']
Unique values in column SaleCondition: ['Normal' 'Abnorml' 'Partial' 'AdjLand' 'Alloca' 'Family']
```
**HeatMap of given dataset**
---
<img width="1110" alt="Screenshot 2025-03-15 at 7 43 23 PM" src="https://github.com/user-attachments/assets/f8469b4c-0e0d-4085-816b-ab4b77f7fb89" />

**Applying Ordinal encoding techniques for some of the Caterogical data columns**
---
```
# Applying Ordinal encoding techniques for some of the Caterogical data columns
# Unique values in column Street: ['Pave' 'Grvl']
Street_map = {"Grvl": 0, "Pave": 1}
rdf['Street'] = df['Street'].map(Street_map)

# Unique values in column LotShape: ['Reg' 'IR1' 'IR2' 'IR3']
LotShape_map = {"IR3": 0, "IR2": 1, "IR1": 2, "Reg": 3}
rdf['LotShape'] = rdf['LotShape'].map(LotShape_map)

# Unique values in column LandContour: ['Lvl' 'Bnk' 'Low' 'HLS']
LandContour_map = {"Low": 0, "Lvl": 1, "Bnk": 2, "HLS": 3}
rdf['LandContour'] = rdf['LandContour'].map(LandContour_map)

# Unique values in column Utilities: ['AllPub' 'NoSeWa']
Utilities_map = {"NoSeWa": 0, "AllPub": 1}
rdf['Utilities'] = rdf['Utilities'].map(Utilities_map)

# Unique values in column LotConfig: ['Inside' 'FR2' 'Corner' 'CulDSac' 'FR3']
LotConfig_map = {"Inside": 0, "FR2": 1, "FR3": 2, "Corner":3, "CulDSac":4}
rdf['LotConfig'] = rdf['LotConfig'].map(Utilities_map)
```

**HeatMap of given dataset with categorical columns**
---
<img width="1110" alt="Screenshot 2025-03-15 at 7 47 22 PM" src="https://github.com/user-attachments/assets/efa9032d-4ba0-4348-a225-e23927c750af" />

**Histplot of given dataset with Numerical columns**
---

```
import math
sns.set_style('darkgrid')
numerical_columns_ccdf = df.select_dtypes(['int64', 'float64']).columns
np.seterr(divide='ignore', invalid='ignore')
total_plots = len(numerical_columns_ccdf)
rows = math.ceil(total_plots / 4)
cols = 4

plt.figure(figsize=(20,20))
for index, feature in enumerate(numerical_columns_ccdf):
    plt.subplot(rows, cols, index + 1)
    feature_data = np.where(df[feature] == 0, np.log(df[feature] + 0.5), np.log(df[feature]))
    # Plotting the histogram with KDE
    sns.histplot(feature_data, kde=True, color='g')
    plt.xlabel(feature)
    plt.ylabel('distribution')
    plt.title(f"{feature} distribution")

plt.tight_layout()
```
<img width="1110" alt="Screenshot 2025-03-15 at 7 49 29 PM" src="https://github.com/user-attachments/assets/57d6cf5b-00ee-4475-b7a7-e4499c873dfa" />

**Eliminating Outliers using IRQ**
---
It was observed that if we use IRQ to emilinate outliers, we are almost loosing 95% of the given data and hence used all the given data for modelling.
```
irq = df['SalePrice'].quantile(.75) - df['SalePrice'].quantile(.25)
lower_bound = df['SalePrice'].quantile(.25) - 1.5 * irq
upper_bound = df['SalePrice'].quantile(.75) + 1.5 * irq
df_irq = df[(df['SalePrice'] > lower_bound) & (df['SalePrice'] > upper_bound)].copy()
irq_data_lost = 1 - (df_irq.shape[0]/df.shape[0])
print("We lost {:.2%} of the data by the IRQ method" .format(irq_data_lost))
    >>> We lost 95.82% of the data by the IRQ method
```
**Scatter Plot of Sale Data**
---
<img width="1110" alt="Screenshot 2025-03-15 at 8 12 46 PM" src="https://github.com/user-attachments/assets/fda8c31d-9a82-48be-ba1e-992f145a8834" />

**Boxplot of Scale Price Log vs Building Type**
---
<img width="1110" alt="Screenshot 2025-03-15 at 8 15 12 PM" src="https://github.com/user-attachments/assets/493f4c3f-c49f-4d2b-98c2-d21dc7d92f31" />

**Violin Plot of Sale Price Log vs Exterior Condition**
---
<img width="1110" alt="Screenshot 2025-03-15 at 8 17 22 PM" src="https://github.com/user-attachments/assets/f43bbfed-470f-4ddf-9229-f8cfd58afad7" />


**Model Comparisons:**
---
<img width="1110" alt="Screenshot 2025-03-15 at 8 43 44 PM" src="https://github.com/user-attachments/assets/36ce8946-a630-49cd-a9a3-a66c2279bf9f" />

*Best performance on Test Data: The Decision Tree (DT) and Random Forest (RF) models perform best in terms of both low errors (MAE and MSE) and high R² values

*Best R² Score on Test: Decision Tree (DT) and Random Forest (RF)

*Most Time-Efficient: K-Nearest Neighbors (KNN), but its performance is not as strong as others

*Worst performance: Support Vector Machine (SVM) is performing poorly across all metrics

*It seems DT, RF, and XGB are the most reliable models for this dataset, with DT being particularly strong at both fitting and generalization, albeit with longer training times for RF and XGB


<img width="1110" alt="Screenshot 2025-03-15 at 8 49 22 PM" src="https://github.com/user-attachments/assets/c55f87c8-4d2f-477b-9e85-c12a3c977fda" />


**Summary of Insights**
---
*Best Test Accuracy: Decision Tree (DT), with an accuracy of 38.63%, but it is likely overfitting due to the perfect training accuracy

*Best Train Accuracy: Decision Tree (DT) (100%) performs excellently on training data, but that may not be useful in practice if it overfits

*Most Time-Efficient: K-Nearest Neighbors (KNN), with a very low runtime of just 0.0042 seconds, although the performance is poor

*Worst Test Accuracy: Logistic Regression (LogR) and Support Vector Machine (SVM) both perform poorly on the test set, with very low accuracy (around 0.0027 to 0.0164)

*Slowest Model: Logistic Regression (LogR) takes the longest time to train, with over 184 seconds, despite its relatively high accuracy on the training data


**Recommendation**
---
*Decision Tree (DT) stands out as the best among the models, despite the overfitting.

*KNN is very fast but not effective for this dataset.

*Logistic Regression (LogR) and SVM should likely be reconsidered, as they both show poor test accuracy and relatively slow performance for this problem.



