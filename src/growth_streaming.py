import os
import shutil

from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession
from pyspark.sql.functions import (avg, col, coalesce, count, date_format, 
                                   least, lit, sum as spark_sum, to_timestamp, when, window)
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType


spark = SparkSession.builder.appName("MortgageStressTestingStreaming").getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

MODELING_INPUT_PATH = "data/processed/modeling_base.parquet"
FRED_INPUT_PATH = "data/processed/fred_monthly.parquet"
MODEL_INPUT_PATH = "models/pd_logistic_regression"
STREAM_INPUT_PATH = "data/stream/incoming"
STREAM_OUTPUT_PATH = "outputs/streaming"
CHECKPOINT_PATH = "checkpoints/streaming_growth"

# UPB is used as EAD, the dollar balance exposed to default.
# LGD is estimated as a percentage using a simple LTV-based proxy.
STRESS_LGD_ADD_ON = 0.10
MAX_STRESS_LGD = 0.75


STREAM_SCHEMA = StructType([
    StructField("loan_id", StringType(), True),
    StructField("event_time", StringType(), True),
    StructField("channel", StringType(), True),
    StructField("first_time_homebuyer_flag", StringType(), True),
    StructField("loan_purpose", StringType(), True),
    StructField("property_type", StringType(), True),
    StructField("occupancy_status", StringType(), True),
    StructField("property_state", StringType(), True),
    StructField("credit_score_band", StringType(), True),
    StructField("ltv_band", StringType(), True),
    StructField("dti_band", StringType(), True),
    StructField("loan_age_band", StringType(), True),
    StructField("upb_band", StringType(), True),
    StructField("current_interest_rate", DoubleType(), True),
    StructField("original_interest_rate", DoubleType(), True),
    StructField("original_upb", DoubleType(), True),
    StructField("current_actual_upb", DoubleType(), True),
    StructField("original_loan_term", IntegerType(), True),
    StructField("loan_age", IntegerType(), True),
    StructField("remaining_months_to_maturity", IntegerType(), True),
    StructField("original_ltv", DoubleType(), True),
    StructField("original_cltv", DoubleType(), True),
    StructField("number_of_borrowers", IntegerType(), True),
    StructField("dti", DoubleType(), True),
    StructField("borrower_credit_score", IntegerType(), True)])


def reset_stream_folders() -> None:
    for path in [STREAM_INPUT_PATH, STREAM_OUTPUT_PATH, CHECKPOINT_PATH]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)


def create_streaming_events():
    source_df = spark.read.parquet(MODELING_INPUT_PATH)

    # Pick one recent record per loan so the stream behaves like new loans being added.
    selected_events = source_df.select("loan_id", "channel", "first_time_homebuyer_flag", "loan_purpose",
                                        "property_type", "occupancy_status", "property_state",
                                        "credit_score_band", "ltv_band", "dti_band", "loan_age_band",
                                        "upb_band", "current_interest_rate", "original_interest_rate",
                                        "original_upb", "current_actual_upb", "original_loan_term",
                                        "loan_age", "remaining_months_to_maturity", "original_ltv",
                                        "original_cltv", "number_of_borrowers", "dti", "borrower_credit_score")\
                                            .limit(30).withColumn("event_time", lit("2025-04-01 09:00:00"))
    

    selected_events.coalesce(1).write.mode("overwrite").json(STREAM_INPUT_PATH)
    print(f"Streaming input events written to {STREAM_INPUT_PATH}")


def score_streaming_loans():
    model = PipelineModel.load(MODEL_INPUT_PATH)
    fred_df = spark.read.parquet(FRED_INPUT_PATH)

    stream_df = spark.readStream.schema(STREAM_SCHEMA).json(STREAM_INPUT_PATH)\
                     .withColumn("event_timestamp", to_timestamp(col("event_time")))\
                     .withColumn("reporting_month", date_format(col("event_timestamp"), "yyyy-MM"))
    

    joined_stream = stream_df.join(fred_df, on="reporting_month", how="left")

    # QUESTION ANSWERED:
    # As new loans arrive, what is their predicted risk band and expected loss?
    predicted_df =  model.transform(joined_stream)\
                          .withColumn("pd", vector_to_array(col("probability"))[1])\
                          .withColumn("pd_band", when(col("pd") < 0.01, "A: <1%")\
                                                .when(col("pd") < 0.03, "B: 1%-3%")
                                                .when(col("pd") < 0.07, "C: 3%-7%")
                                                .when(col("pd") < 0.15, "D: 7%-15%")
                                                .otherwise("E: 15%+"))\
                          .withColumn("ead", coalesce(col("current_actual_upb"), col("original_upb"), lit(0.0)))\
                          .withColumn("lgd", when(col("original_ltv").isNull(), lit(0.35))\
                                            .when(col("original_ltv") <= 60, lit(0.20))
                                            .when(col("original_ltv") <= 80, lit(0.30))
                                            .when(col("original_ltv") <= 90, lit(0.40))
                                            .when(col("original_ltv") <= 97, lit(0.50))
                                            .otherwise(lit(0.60)))\
                          .withColumn("expected_loss", col("pd") * col("lgd") * col("ead"))

    # QUESTION ANSWERED:
    # At the streaming-batch level, how many new loans are in each risk band,
    # and how much total expected loss do they add to the growing portfolio?
    aggregated = predicted_df\
        .withWatermark("event_timestamp", "1 minute")\
        .groupBy(window(col("event_timestamp"), "1 minute"), col("pd_band"))\
        .agg(count("*").alias("new_loan_count"), avg("pd").alias("avg_pd"), 
             spark_sum("ead").alias("total_ead"),
             spark_sum("expected_loss").alias("total_expected_loss"))

    def write_batch(batch_df, batch_id: int) -> None:
        print(f"Streaming batch {batch_id}")
        batch_df.show(truncate=False)
        batch_df.coalesce(1).write.mode("overwrite").option("header", True).csv(
            os.path.join(STREAM_OUTPUT_PATH, f"batch_{batch_id}"))

    query = aggregated.writeStream.foreachBatch(write_batch).outputMode("complete")\
             .option("checkpointLocation", CHECKPOINT_PATH).trigger(availableNow=True)\
                .start()

    query.awaitTermination()
    print(f"Streaming outputs written to {STREAM_OUTPUT_PATH}")


def main() -> None:
    reset_stream_folders()
    create_streaming_events(spark)
    score_streaming_loans(spark)
    print("Streaming simulation complete.")
    spark.stop()

if __name__ == "__main__":
    main()
