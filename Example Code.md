---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region -->
# Description

Let us take finite engine as example. Feature Engine is a good library for feature engineering like imputation, targe encoding, etc. 

Sometime, we need to add in some small code in between two feature tranformations. This can be done seamlesly in notebook. However, it requires extra work if we would like to bring the code into scikit learn pipeline and serialize it (with pickle) into production, unless you make those code into objects, which requires some extra works. 

As to mitigate this, we have introduced "Paip", a very simple pipeline module that allows us to insert small piece of code in the middle of pipeline.

In the following demo, we will impute the titanic data set with Feature Engine. We use Paip to create the pipeline and it allows to insert a small piece of code in between the imputers.


Description of titanic data can be found here. https://www.kaggle.com/c/titanic/data
<!-- #endregion -->

# Imports 

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from feature_engine import categorical_encoders as ce
import feature_engine.missing_data_imputers as mdi

from Paip.Paip import Paip


```

# Retrieve Data

```python
#
# We read the data from the Internet directly.
# And, we directly replace those data with "?" into np.nan.
#
url = 'https://www.openml.org/data/get_csv/16826755/phpMYEkMl'
data = pd.read_csv(url, na_values = '?')

```

```python
data.sample(3)
```

```python
x_col_list = [
    'pclass',
    'sex',
    'age',
    'sibsp',
    'parch', 
    'embarked'
]

y_col = 'survived'

X = data[x_col_list]
y = data[y_col]
```

```python
X.sample(3)
```

```python
# Percentage of missing values
X.isnull().sum(axis = 0) / X.shape[0]
```

```python
X.dtypes
```

```python
# Separate into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
```

```python

```

# Numerical Imputation

```python
# Imputation on numerical values
num_imputer = mdi.MeanMedianImputer(imputation_method='median',
                                       variables=['age', 'sibsp', 'parch'] )

# fit the imputer
num_imputer.fit(X_train)

# transform the data
train_t= num_imputer.transform(X_train)
test_t= num_imputer.transform(X_test)

print( num_imputer.imputer_dict_ )
```

```python
# Add to pipeline

step_list = []

step_list.append(
    {'name': 'numerical imputer',
     'obj_dict' : {'num_imputer': num_imputer},
      'run_dict':{
                'transform': ['X = num_imputer.transform(X)'],
            },
     'output_list' : ['X', ]
    }
)


```

```python
train_t.dtypes
```

# Categorical Imputation

```python
# Imputation on categorical values

# Required
cat_variables=['pclass', 'sex', 'embarked']
train_t[cat_variables] =  train_t[cat_variables].astype('O')


cat_imputer = mdi.CategoricalVariableImputer(variables=cat_variables)

# fit the imputer
cat_imputer.fit(train_t)

# transform the data
train_t= cat_imputer.transform(train_t)
test_t= cat_imputer.transform(test_t)

print( cat_imputer.imputer_dict_ )
```

```python
train_t.dtypes
```

```python
# Add to pipeline

# To transform the variable to categorical.
transform_code ='''
cat_variables=['pclass', 'sex']
X[cat_variables] =  X[cat_variables].astype('O')
'''

step_list.append(
    {'name': 'categorical imputer',
     'obj_dict' : {'cat_imputer': cat_imputer},
      'run_dict':{
                'transform': [transform_code, 
                              'X = cat_imputer.transform(X)'],
            },
     'output_list' : ['X', ]
    }
)



```

# Test Tranform with Pipeline

```python
paip1 = Paip(step_list)
```

```python
paip1.__dict__
```

```python
paip1.run('transform',    # run those command that tag with "transform"
          {'X': X_test,}, # the variable to bring into the pipeline.
          debug = True)   # Debug output 
```

```python
# The transformed output is obtained here.
X_test_tr = paip1.output_dict['X']

```

```python
X_test_tr
```

```python

```
