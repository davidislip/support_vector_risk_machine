# Integrating SVM and MVO

Welcome to the supporting code for integrating support vector machines and capital allocation. 

## Data 

For licensing reasons with WRDS and Tiingo, we cannot share the raw data. To get started you will want to create your own database that contains all the financial data you will need. We use SQLLITE and have supplementary scripts located under the database folder to pull and store from a variety of sources. Please follow the instructions there to obtain all the data you require. 

After that, apply the data processing routines in notebooks 
* ```1. Data_S&P.ipynb``` and ```2. Data_ETF.ipynb```

## Experiments
 
```Experiments.ipynb``` runs the financial performance experiments for the Card-MVO (cardinality-constrained optimization) and the SVM-MVO model. Instructions are contained in the notebook.

The ```SingleInstance.ipynb``` notebook produces the plots for the single instance demonstration in Section 4.1 of the paper. 

The ```Results.ipynb``` notebook analyzes the output of the Experiments.

The ```Results Gaps and Times.ipynb``` notebook analyzes the runtimes and optimality gaps.
