##1. Fit decision tree in SAS##
The decision tree is conducted by PROC HPSPLIT in SAS. To build the decision tree on training and testing data, we first randomly shuffle the original data and select the first 700 observations as training data and the rest as testing data. 
The SAS code is as follows.
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

PROC HPSPLIT DATA = credit_train SEED = 123;
TITLE 'Decision tree for credit training data';
CLASS checking_balance credit_history purpose savings_balance 
	  employment_duration other_credit housing job phone default;
MODEL default(event = 'yes') = checking_balance months_loan_duration  
		credit_history	purpose amount savings_balance employment_duration
	   	percent_of_income years_at_residence age other_credit 
	   	housing existing_loans_count job dependents phone default;
GROW ENTROPY;
PRUNE COSTCOMPLEXITY;
CODE FILE = '/home/sujay/Practice/dt_credit.sas';
RUN;

TITLE 'Predictions on credit testing data';
DATA credit_pred(KEEP = Actual Predicted);
SET credit_test END = EOF;
%INCLUDE "/home/sujay/Practice/dt_credit.sas";
Actual = default;
Predicted = (P_defaultyes >= 0.5);
run;

TITLE "Confusion Matrix Based on Cutoff Value of 0.5";
PROC FREQ DATA = credit_pred;
TABLES Actual*Predicted /norow nocol nopct;
RUN;
```

![cost_comp_sas](https://cloud.githubusercontent.com/assets/16762941/12804239/5286cfb8-cabf-11e5-9aee-8a490e5bbf1a.png)

The trend of cost complexity analysis shows that the smallest average ASE (0.176) obtains at cost complexity parameter = 0.0068. Let's look at the graph of fitted tree as follows. We can see that the most four important features are checking_balance, month_loan_duration, credit_history and savings_balance. To interpret the tree, we see that if the checking_balance is greater than 200 DM or unknown, 318 samples are classified as 'no' and 87.74% of them are truly 'no' in the training data. Otherwise, if the checking_balance is smaller or equal to 200 DM, and if months_loan_duration is less than 21.68 and if the credit_history is perfect or very good, 21 samples are classified as 'yes' with 71.43% accurate rate. We can interpret others in the same fasion.

![tree_sas](https://cloud.githubusercontent.com/assets/16762941/12804299/fdf6fecc-cabf-11e5-956f-23c8575640e3.png)

Finally, let's check the accuracy of the fitted decision tree on testing data. The confusion matrix gives us the accuracy of 59% ((145 + 32)/300) which is somewhat low. However, the result can be improved by using random forest or gradient boosting that will be covered in the latter section.

![conf_mat_sas](https://cloud.githubusercontent.com/assets/16762941/12804393/da77c714-cac0-11e5-85e4-71659664e8a7.png)


## 2. Fit decision tree in our Code ##
Python `sklearn` package provides numerous functions to perform machine learning methods, including decision tree. We now give the Python code to fit the decision tree for bank loans data. 

```python
import pandas as pd
import os
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics
from sklearn.preprocessing import LabelEncoder
os.chdir('/Users/Sujay/Desktop/Coursera_ML')

credit = pd.read_csv("credit.csv")

credit = credit.dropna()
targets = LabelEncoder().fit_transform(credit['default'])

predictors = credit.ix[:,credit.columns != 'default']

# Recode categorical variables as numeric variables
predictors.dtypes
for i in range(0,len(predictors.dtypes)):
    if predictors.dtypes[i] != 'int64':
        predictors[predictors.columns[i]] = LabelEncoder().fit_transform(predictors[predictors.columns[i]])

pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, targets, test_size=.4)

#Build model on training data
classifier = DecisionTreeClassifier().fit(pred_train,tar_train)
predictions = classifier.predict(pred_test)

sklearn.metrics.confusion_matrix(tar_test,predictions)
sklearn.metrics.accuracy_score(tar_test, predictions)

#Displaying the decision tree
from sklearn import tree
#from StringIO import StringIO
from io import StringIO
#from StringIO import StringIO 
from IPython.display import Image
out = StringIO()
tree.export_graphviz(classifier, out_file=out)
import pydotplus
graph=pydotplus.graph_from_dot_data(out.getvalue())
Image(graph.create_png())
```
Since Python does not provide pruing on the decision tree, the classification accuracy (69%) may be higher than that from SAS. Also, it results in a huge tree shown in the following graph. Without pruning, the tree is more likely to overfit the data.

```python
>>> sklearn.metrics.confusion_matrix(tar_test,predictions)
Out[2]: 
array([[220,  66],
       [ 58,  56]])

>>> sklearn.metrics.accuracy_score(tar_test, predictions)
Out[3]: 0.68999999999999995
```

![tree_python](https://cloud.githubusercontent.com/assets/16762941/12804810/397d63c4-cac4-11e5-8a9e-259b8f45391a.png)
