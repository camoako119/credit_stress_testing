from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when




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
selected_df = df.select(col("_c1").alias("loan_id"),
                        col("_c2").alias("monthly_reporting_period"),
                        col("_c3").alias("channel"),
                        col("_c7").cast("double").alias("current_interest_rate"),
                        col("_c8").cast("double").alias("original_interest_rate"),
                        col("_c9").cast("double").alias("current_actual_upb"),
                        col("_c12").cast("int").alias("original_loan_term"),
                        col("_c13").alias("origination_date"),
                        col("_c14").alias("first_payment_date"),
                        col("_c15").cast("int").alias("loan_age"),
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


# taking a look at the values in the delinquency column
selected_df.select("current_loan_delinquency").distinct().show()


# When a loan is past 90 days, it is considered seriously delinquent
# Creating a flag column called seriously_delinquent for such loans
selected_df = selected_df.withColumn("is_seriously_delinquent", 
                                     when(col("current_loan_delinquency") >= 3, 1)\
                                        .otherwise(0))


print("Schema:")
selected_df.printSchema()

print()

print("Sample Records:")
selected_df.show(10, truncate=False)

print()

print("Count of delinquency for each month:")
selected_df.groupBy("current_loan_delinquency")\
    .count().orderBy("current_loan_delinquency").show()

print("Portfolio by State:")
selected_df.groupBy("property_state").count().orderBy("property_state").show()
print()

selected_df.write.mode("overwrite").parquet("outputs/cleaned_mortgage_sample")




