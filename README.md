# support_vector_risk_machine
 
Welcome to the supporting code for integrating support vector machines and capital allocation. 

For licensing reasons with WRDS and Tiingo, we cannot share the raw data. To get started you will want to create your own database that contains all the financial data you will need. We use SQLLITE and have supplementary scripts located under the database folder to pull and store from a variety of sources. Please follow the instructions there to obtain all the data you require. 

After that, apply the data processing routines in notebooks 1. Data_S&P and 2. Data_ETF

The Experiments notebook showcases how to run the Card-MVO (cardinality-constrained optimization) and the SVM-MVO model.

The SingleInstance notebook produces the plots for the single instances demonstration

The Results notebook analyzes the output of the Experiments

The Results Gaps and Times notebook analyzes the runtimes and optimality gaps.
