# What is Paip? And Why Paip?
"Paip" is a Malay word of "pipe", a lightweight library that is alternative to Scikit Learn pipeline. 

In general, Scikit Learn's pipeline facilitates the chaining of transformers in series and execute them in sequential order. Then, the pipeline can be serialized and re-used easily at different environment. However, the existing Scikit Learn's pipeline only allows objects to be loaded into the step list. If there is a small piece of code that is needed in between transformers (e.g. change of column name of data frame), we need to revise the small piece of code into object-oriented paradigm in order to obtain the corresponding objects. Need less to say, the class definition has to be imported at the new environment during the unpicke process. 

To overcome this issue, we have developed a lightweight library named Paip that allows both runtime code and transformers to be executed in the pipelines.

For example, we have a data frame, `X` that needs to go through two feature transformations, i.e. `transformer1` and `transformer2`. In between the transformers, we need to rename the column, "x1" to "x1_new". The code is presented in the following.

```python
from paip import paip

# Somewhere here, we have created the data frame, `X`.
# Somewhere here, we have created feature transformation, named `transformer1`.
# Somewhere here, we have created feature transformation, named `transformer2`.

step_list = []

# Add the transformer 1 into the step list.
step_list.append(
    {'name': 'Transformer 1', # name
     'obj_dict' : {'transformer1': transformer1}, # The objects to be serialized in pickle.
      'run_dict':{
                'transform': ['X = transformer1.transform(X)'], # Run this code.
            },
     'output_list' : ['X', ] # Whitelist these variables will go to next step.
    }
)

# Add the transformer 2 into the step list.
step_list.append(
    {'name': 'Transformer 1',
     'obj_dict' : {'transformer1': transformer1},
      'run_dict':{
                'transform': [
                    'X = X.rename(columns={"x1": "x1_new")',  # Rename column.
                    'X = transformer1.transform(X)'
                ],
            },
     'output_list' : ['X', ]
    }
)


paip1 = Paip(step_list)

paip1.run('transform', # run those command that tag with "transform"
          {'X': X,},    # the variable to bring into the pipeline.
          debug = True)   # Debug output 

# Retrieve the output
X_out = paip1.output_dict['X']

```

And to serialize Paip, we use the existing joblib.
```
import joblib
joblib('./test.paip', paip1)

```
