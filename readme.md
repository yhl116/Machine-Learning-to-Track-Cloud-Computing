# Machine Learning to Track Cloud Computing
This is a MEng Electrical & Electronics Engineering Final Year Project for Imperial College London. It outlines a correlation based one-step forecasting model for cloud resource utilisation.

### Dataset
The dataset used is Google trace data (version 2) which consists of measures of of CPU utilisation and memory utilisation of 12,476 machines. Each machine is sampled for about 29 days at 5 minutes interval. The total number of samples per machine in ...

### Proposed Framework Overview
The proposed correlation based framework consists of four parts:
1. __Dynamic time-series clustering__ using a rolling correlation.
2. __Computation of general model__ for each cluster. The general model maximises correlation with other time-series in the cluster.
3. __Temporal forecasting__ on the general model using ARIMA models.
4. __Forecast scaling__ to scale the forecasted value on the general model to fit all the other time-series in the cluter.

### Validation
The proposed framework is benchmarked against the following frameworks:
1. An MSE based forecasting framework proposed in the paper Online Collection and Forecasting of Resource Utilization in Large-Scale Distributed Systems
2. A brute force method where we create a single ARIMA model for each time-series
3. A naive method where we use a lagged version of the pervious time-step as the forecasted value.

### Directory
1. All the analysis are done in the `analysis` folder
2. All the frameworks are individually designed and tested in each of the file in the `implementation` folder.
3. `implementation//all_frameworks.py` is a compilation of the frameworks
4. Validation of the framework is done in `implementation//validation.ipynb`
5. Refer to 'Final Report.pdf' for all the design choices and for more indepth explanation of the framework