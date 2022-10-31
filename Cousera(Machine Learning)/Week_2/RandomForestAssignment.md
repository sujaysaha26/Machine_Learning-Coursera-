In this assignment, we use a finanial dataset which contains information on loans obtained from a credit agency in Germany to perform the random forest in order to identify risky bank loans. There are total 1000 observations and 17 features in this dataset. The **target** is "default" which is a binary variable: 'yes' and 'no', meaning whether the loan went into default. 

The **explanatory variables** consist of the following 16 components:
- checking_balance (categorical)  : "< 0 DM",     "> 200 DM",   "1 - 200 DM", "unknown"
- months_loan_duration (interval) : 4 - 72
- credit_history (categorical)    : "critical",  "good",   "perfect",   "poor",      "very good"
- purpose  (categorical)          : "business",   "car",   "car0",      "education",  "furniture/appliances", "renovations" 
- amount   (interval)             : 250 - 18424
- savings_balance (categorical)   : "< 100 DM",     "> 1000 DM",     "100 - 500 DM",  "500 - 1000 DM", "unknown" 
- employment_duration (categorical): "< 1 year",    "> 7 years",   "1 - 4 years", "4 - 7 years", "unemployed" 
- percent_of_income (interval)    : 1 - 4
- years_at_residence (interval)   : 1 - 4
- age                (interval)   : 19 - 75
- other_credit     (categorical)  :  "bank",  "none",  "store"
- housing          (categorical)  : "other", "own",   "rent"
- existing_loans_count (interval) : 1 - 4
- job             (categorical)   : "management", "skilled",    "unemployed", "unskilled"
- dependents   (interval)         : 1 - 2
- phone        (categorical)      : "no",  "yes"

## Run random forest in SAS ##
We use HPFOREST procedure in SAS to run the random forest which builds many decision trees ranther than a single decision tree in order to improve the accuracy of prediction. In the fitting procedure, we first randomly split the entire data into training set with 700 observations and testing set with 300 observations. We then run the random forest on the training data and make predictions on testing data using HP4SCORE procedure. 
```
TITLE 'Import credit.csv data';
FILENAME CSV "/home/sujay/Practice/credit.csv" TERMSTR = CRLF;
PROC IMPORT DATAFILE = CSV OUT = credit DBMS = CSV REPLACE;
RUN;

PROC PRINT DATA = credit(OBS = 10); 
RUN;

TITLE 'Create training and testing data respectively by randomly shuffling strategy';
PROC SQL;
CREATE TABLE credit AS
SELECT * FROM credit
ORDER BY ranuni(0)
;
RUN;
TITLE 'Training data with 700 observations';
DATA credit_train;
SET credit;
IF _N_ <= 700 THEN OUTPUT;
RUN;
TITLE 'Testing data with 300 observations';
DATA credit_test;
SET credit;
IF _N_ > 700 THEN OUTPUT;
RUN;

ODS GRAPHICS ON;
PROC HPFOREST DATA = credit_train;
TITLE 'Random forest for credit training data';
TARGET default/LEVEL = BINARY;
INPUT checking_balance credit_history purpose savings_balance 
	  employment_duration other_credit housing job/ LEVEL= NOMINAL;
INPUT phone/LEVEL=BINARY;
INPUT months_loan_duration amount percent_of_income years_at_residence 
	  age existing_loans_count dependents/LEVEL = INTERVAL;
SAVE FILE = '/home/sujay/Practice/rf_credit.sas';
RUN;

PROC HP4SCORE DATA = credit_test;
TITLE 'Predictions on credit testing data';
ID default;
SCORE FILE = '/home/sujay/Practice/rf_credit.sas' OUT = rfscore;
RUN;

TITLE "Confusion matrix for testing data";
PROC FREQ DATA = rfscore;
TABLES default*I_default /norow nocol nopct;
RUN;
```

## Code ##

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import sklearn.metrics
from sklearn.preprocessing import LabelEncoder
import matplotlib.pylab as plt

credit = pd.read_csv("credit.txt",sep = "\t")

credit = credit.dropna()
targets = LabelEncoder().fit_transform(credit['default'])

predictors = credit.ix[:,credit.columns != 'default']

# Recode categorical variables as numeric variables
predictors.dtypes
for i in range(0,len(predictors.dtypes)):
    if predictors.dtypes[i] != 'int64':
        predictors[predictors.columns[i]] = LabelEncoder().fit_transform(predictors[predictors.columns[i]])

pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, targets, test_size=.3)

# Build model on training data
classifier = RandomForestClassifier(n_estimators = 25)
classifier = classifier.fit(pred_train, tar_train)

# Make predictions on testing data
predictions = classifier.predict(pred_test)

# Calculate accuracy
sklearn.metrics.confusion_matrix(tar_test, predictions)
sklearn.metrics.accuracy_score(tar_test, predictions)

# Fit an extra trees model to the training data 
model = ExtraTreesClassifier().fit(pred_train, tar_train)
# Display the relative importance of each attribute
print(pd.Series(model.feature_importances_, index = predictors.columns).sort_values(ascending = False))

"""
Running a different number of trees and see the effect
 of that on the accuracy of the prediction
"""

ntree = [50,150,250,350,450,550,650,750,850,950,1000]
accuracy = []

for idx in range(len(ntree)):
    classifier = RandomForestClassifier(n_estimators = ntree[idx])
    classifier = classifier.fit(pred_train, tar_train)
    predictions = classifier.predict(pred_test)
    accuracy.append(sklearn.metrics.accuracy_score(tar_test,predictions))

pd.Series(accuracy, index = ntree).sort_values(ascending = False)

plt.plot(ntree,accuracy)
plt.show()

```
In the above procedure, We first build a random forest with 25 decision trees. This gives us 74% accuracy on the testing data. 
```python
# Calculate accuracy
>>> sklearn.metrics.confusion_matrix(tar_test, predictions)
Out[39]: 
array([[189,  32],
       [ 46,  33]])

>>> sklearn.metrics.accuracy_score(tar_test, predictions)
Out[40]: 0.73999999999999999
```
We also explore the importane of 16 explanatory variables. The first three most important explanatory variables are checking_balance, amount, months_loan_duration which are slightly different from those obtained in R. 
```python
# Display the relative importance of each attribute
>>> print(pd.Series(model.feature_importances_, index = predictors.columns).sort_values(ascending = False))
checking_balance        0.133015
amount                  0.109541
months_loan_duration    0.096196
age                     0.086818
employment_duration     0.064515
credit_history          0.064045
percent_of_income       0.063428
purpose                 0.063158
savings_balance         0.055704
years_at_residence      0.052617
job                     0.045315
existing_loans_count    0.039384
other_credit            0.038604
housing                 0.035119
phone                   0.030843
dependents              0.021698
dtype: float64
```

```python
>>> pd.Series(accuracy, index = ntree).sort_values(ascending = False)
Out[43]: 
850     0.760000
250     0.760000
1000    0.756667
950     0.756667
750     0.756667
650     0.756667
450     0.756667
350     0.756667
550     0.743333
50      0.740000
150     0.736667
dtype: float64
```
![accuracy_p](https://cloud.githubusercontent.com/assets/16762941/12930934/23526c6e-cf49-11e5-9a11-de2044197afc.png)

