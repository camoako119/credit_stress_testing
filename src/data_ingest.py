import glob
import os
from functools import reduce

from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, count, date_format, to_date, when

spark = SparkSession.builder.appName("MortgageStressTestingEDA").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

df = spark.read.option("delimiter", "|").option("header", "false").csv("data/*.csv")

# Drop first column because it is empty
df = df.drop("_c0")


print("Rows:", df.count())
print("Columns:", len(df.columns))

# I am currently looking at these columns from the data as there are 108
# columns. I will add some more columns as I progress on the project
# Comverting data types

# Renamed the mortgage data using a PDF provided by Fannie Mae and 
# casting the data types 
selected_df = df.select(col("_c1").alias("loan_id"),
                        col("_c2").alias("monthly_reporting_period"),
                        col("_c3").alias("channel"),
                        col("_c4").alias("seller_name"),
                        col("_c5").alias("servicer_name"),
                        col("_c7").cast("double").alias("current_interest_rate"),
                        col("_c8").cast("double").alias("original_interest_rate"),
                        col("_c9").cast("double").alias("original_upb"),
                        col("_c11").cast("double").alias("current_actual_upb"),
                        col("_c12").cast("int").alias("original_loan_term"),
                        col("_c13").alias("origination_date"),
                        col("_c14").alias("first_payment_date"),
                        col("_c15").cast("int").alias("loan_age"),
                        col("_c17").cast("int").alias("remaining_months_to_maturity"),
                        col("_c18").alias("maturity_date"),
                        col("_c19").cast("double").alias("original_ltv"),
                        col("_c20").cast("double").alias("original_cltv"),
                        col("_c21").cast("int").alias("number_of_borrowers"),
                        col("_c22").cast("double").alias("dti"),
                        col("_c23").cast("int").alias("borrower_credit_score"),
                        col("_c25").alias("first_time_homebuyer_flag"),
                        col("_c26").alias("loan_purpose"),
                        col("_c27").alias("property_type"),
                        col("_c29").alias("occupancy_status"),
                        col("_c30").alias("property_state"),
                        col("_c39").cast("int").alias("current_loan_delinquency"))


# selected_df.select("maturity_date").distinct().orderBy("current_loan_delinquency").show()


# Creating my own columns that I will use in the modelling and EDA process
mortgage_df = selected_df.withColumn("is_seriously_delinquent", 
                                     when(col("current_loan_delinquency") >= 3, 1)\
                                        .otherwise(0))\
                        .withColumn("reporting_date", to_date(col("monthly_reporting_period"), "MMyyyy")) \
                        .withColumn("reporting_month", date_format(col("reporting_date"), "yyyy-MM"))\
                        .withColumn("credit_score_band", 
                               when(col("borrower_credit_score").isNull(), "Missing")
                              .when((col("borrower_credit_score") >= 0) & (col("borrower_credit_score") < 620), "Poor")
                              .when((col("borrower_credit_score") >= 620) & (col("borrower_credit_score") < 680), "Fair")
                              .when((col("borrower_credit_score") >= 680) & (col("borrower_credit_score") < 720), "Good")
                              .when((col("borrower_credit_score") >= 720) & (col("borrower_credit_score") < 760), "Very Good")
                              .when(col("borrower_credit_score") >= 760, "Excellent")
                              .otherwise("Missing"))\
                        .withColumn("ltv_band",
                              when(col("original_ltv").isNull(), "Missing")
                             .when(col("original_ltv") <= 60, "<=60")
                             .when(col("original_ltv") <= 80, "61-80")
                             .when(col("original_ltv") <= 90, "81-90")
                             .when(col("original_ltv") <= 97, "91-97")
                             .otherwise(">97"))\
                        .withColumn("dti_band", when(col("dti").isNull(), "Missing")
                             .when(col("dti") <= 20, "<=20")
                             .when(col("dti") <= 35, "21-35")
                             .when(col("dti") <= 45, "36-45")
                             .otherwise("46+"))\
                        .withColumn("loan_age_band",when(col("loan_age").isNull(), "Missing")
                             .when(col("loan_age") <= 12, "0-12 months")
                             .when(col("loan_age") <= 36, "13-36 months")
                             .when(col("loan_age") <= 60, "37-60 months")
                             .otherwise("60+ months"))\
                        .withColumn("upb_band",
                            when(col("current_actual_upb").isNull(), "Missing")
                            .when(col("current_actual_upb") < 100000, "<100K")
                            .when(col("current_actual_upb") < 250000, "100K-249K")
                            .when(col("current_actual_upb") < 500000, "250K-499K")
                            .otherwise("500K+"))\
                        .withColumn("risk_state", when(col("current_loan_delinquency").isNull(), "Missing")
                           .when(col("current_loan_delinquency") == 0, "Current")
                           .when(col("current_loan_delinquency") == 1, "30 DPD")
                           .when(col("current_loan_delinquency") == 2, "60 DPD")
                           .otherwise("90+ DPD"))



## Downloading the FRED data and transformiung it to select only 2025 data
## This is so that it aligns with the 2025 Fannie Mae data I have

def download_fred_series(fred_data_path):
    csv_files = glob.glob(os.path.join(fred_data_path, "*.csv"))
    
    data_list = [spark.read.option("header", True)\
            .option("inferSchema", True)\
            .csv(file) for file in csv_files]
    
    joined_df = reduce(lambda x, y : x.join(y, on="observation_date", how="outer"), data_list)

    joined_df = joined_df.withColumnRenamed("observation_date", "date")
    
    return joined_df

fred_df = download_fred_series("data/fred_data")


fred_df = fred_df\
    .withColumn("date", to_date(col("date")))\
    .withColumn("reporting_month", date_format(col("date"), "yyyy-MM"))\
    .groupBy("reporting_month")\
    .agg(
        avg("CSUSHPISA").alias("house_price_index"),
        avg("MORTGAGE30US").alias("mortgage_rate_30yr"),
        avg("UNRATE").alias("unemployment_rate"),
        avg("CPIAUCSL").alias("cpi"),
        avg("FEDFUNDS").alias("fed_funds_rate")).orderBy("reporting_month")

fred_2025 = fred_df.filter(fred_df.reporting_month.like("2025%"))\
    .orderBy("reporting_month")

print("Saving 2025 FRED data")
fred_2025.coalesce(1).write.mode("overwrite").option("header", "true").csv("outputs/fred_2025.csv")


modeling_df = mortgage_df.join(fred_2025, on="reporting_month", how="left")

print("Mortgage data joined with FRED data")

print("Saving the datasets")

# Save the datasets needed by the modeling, EDA process and streaming
mortgage_df.write.mode("overwrite").parquet("data/processed/cleaned_mortgage.parquet")
fred_2025.write.mode("overwrite").parquet("data/processed/fred_monthly.parquet")
modeling_df.write.mode("overwrite").parquet("data/processed/modeling_base.parquet")


print("Data preparation complete.")

