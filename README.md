# Growth-Adjusted Retail Credit Risk Stress Testing

## Member
Cecilia Amoako

## Project Overview
This project will build a retail credit risk stress testing pipeline that models consumer loan portfolio growth through new loan originations. 
The pipeline will predict the probability of default, calculate expected loss, analyze risk migration, and compare baseline and stressed scenarios.

## Chosen Dataset

### Primary Dataset: Fannie Mae Single-Family Loan Performance Data

For the primary dataset, I will use the Fannie Mae Single-Family Loan Performance Data. This dataset contains historical mortgage loan origination records and monthly loan performance records. 

Dataset source: 
https://capitalmarkets.fanniemae.com/credit-risk-transfer/single-family-credit-risk-transfer/fannie-mae-single-family-loan-performance-data


### Secondary Dataset: FRED Macroeconomic Data

For the macroeconomic stress testing component, I will use public time-series variables from available at FRED

Dataset source: https://fred.stlouisfed.org/

Planned macroeconomic indicators include unemployment rate, federal funds rate, mortgage rates, GDP growth, inflation/CPI, and housing price indicators. These variables will be joined to the mortgage loan data by month or quarter and used to create baseline, adverse, and severely adverse stress scenarios.


## Planned Components
- Spark DataFrames for ingestion, cleaning, etc
- Spark SQL for analysis
- Spark Structured Streaming to simulate new loan originations
- Spark MLlib for default prediction, Expected Loss, Transition matrix migration, New loans transition matrix migration
