# Growth-Adjusted Retail Credit Risk Stress Testing

## Member
Cecilia Amoako

## Project Overview
This project will build a retail credit risk stress testing pipeline that models consumer loan portfolio growth through new loan originations. 
The pipeline will predict the probability of default, calculate expected loss, analyze risk migration, and compare baseline and stressed scenarios.

## Proposed Dataset
The project will use consumer loan data from LendingClub and macroeconomic variables from FRED.

## Planned Components
- Spark DataFrames for ingestion, cleaning, feature engineering, and joins
- Spark SQL for analysis
- Spark Structured Streaming to simulate new loan originations
- Spark MLlib for default prediction, Expected Loss, Transition matrix migration, New loans transition matrix migration
