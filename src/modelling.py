
import json
import os

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import Imputer, StringIndexer, VectorAssembler
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (avg, coalesce, col, count, lead,
                                   lit, to_date, sum as spark_sum, when)
from pyspark.ml.functions import vector_to_array

spark = SparkSession.builder.appName("MortgageStressTestingModeling").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

## Load the data needed for modelling. This is the 2025 Q1 - Q4 
## Fannie Mae data that has been joined with FRED data

modeling_df = spark.read.parquet("data/processed/modeling_base.parquet")

## Selecting my numerical variables that I will use
NUMERIC_FEATURES = [
    "current_interest_rate",
    "original_interest_rate",
    "original_upb",
    "current_actual_upb",
    "original_loan_term",
    "loan_age",
    "remaining_months_to_maturity",
    "original_ltv",
    "original_cltv",
    "number_of_borrowers",
    "dti",
    "borrower_credit_score",
    "house_price_index",
    "mortgage_rate_30yr",
    "unemployment_rate",
    "cpi",
    "fed_funds_rate",
]

## Selecting my categorical variables
CATEGORICAL_FEATURES = [
    "channel",
    "first_time_homebuyer_flag",
    "loan_purpose",
    "property_type",
    "occupancy_status",
    "property_state",
    "credit_score_band",
    "ltv_band",
    "dti_band",
    "loan_age_band",
    "upb_band",
]

## This is the label column that will be used as my dependent variable
## The 'y' in the equation
LABEL_COL = "is_seriously_delinquent"

def prepare_training_data(df):
    """
    Clean dataset for modeling by filling in null values
    so that it will not cause errors. 
    """
    ## We filter out rows where is_seriously_delinquent column is NULL
    prepared_df = df.filter(col(LABEL_COL).isNotNull())
    
    # Fill categorical features with "Missing"
    for col_name in CATEGORICAL_FEATURES:
        if col_name in prepared_df.columns:
            prepared_df = prepared_df.withColumn(col_name,
                                                 when(col(col_name).isin("NA", "N/A", "Unknown", ""), "Missing")\
                                                    .otherwise(col(col_name)))
    
    return prepared_df


def build_ml_pipeline():
    """
    QUESTION ANSWERED:
    Which Spark MLlib steps turn raw loan/macro columns into a model-ready feature vector?
    """
    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep",)
                for c in CATEGORICAL_FEATURES]

    imputed_numeric_cols = [f"{c}_imputed" for c in NUMERIC_FEATURES]
    imputer = Imputer(inputCols=NUMERIC_FEATURES, 
                      outputCols=imputed_numeric_cols).setStrategy("mean")

    feature_cols = imputed_numeric_cols + [f"{c}_idx" for c in CATEGORICAL_FEATURES]

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")

    logistic_regression = LogisticRegression(featuresCol="features", labelCol=LABEL_COL,
                                             predictionCol="prediction", probabilityCol="probability")

    return Pipeline(stages=indexers + [imputer, assembler, logistic_regression])


def train_pd_model(df):
    """
    QUESTION ANSWERED:
    Can loan features and macroeconomic features predict serious delinquency risk?
    """
    training_data = prepare_training_data(df)

    training_data = training_data.sample(withReplacement=False, 
                                         fraction=0.1,
                                         seed=50).repartition(64)

    print("Number of records used for ML:")
    print(training_data.count())

    print("Label distribution in ML sample:")
    training_data.groupBy(LABEL_COL).count().show()

    label_count = training_data.select(LABEL_COL).distinct().count()

    if label_count < 2:
        raise ValueError(
            "The sampled training data does not contain both label classes. "
            "We must increase the sample fraction.")

    train_df, test_df = training_data.randomSplit([0.75, 0.25], seed=50)

    pipeline = build_ml_pipeline()
    model = pipeline.fit(train_df)
    predictions = model.transform(test_df)

    # A tiny sample can randomly put only one class in the test split, which makes
    # AUC less meaningful. For that case, evaluate on the full prepared dataset.
    if predictions.select(LABEL_COL).distinct().count() < 2:
        predictions = model.transform(training_data)

    binary_evaluator = BinaryClassificationEvaluator(labelCol=LABEL_COL, 
                                                     rawPredictionCol="rawPrediction", 
                                                     metricName="areaUnderROC")


    accuracy_evaluator = MulticlassClassificationEvaluator(labelCol=LABEL_COL,
                                                            predictionCol="prediction",
                                                            metricName="accuracy")
    
    precision_evaluator = MulticlassClassificationEvaluator(labelCol=LABEL_COL,
                                                             predictionCol="prediction",
                                                             metricName="weightedPrecision")


    metrics = {
        "model_type": "Spark MLlib Logistic Regression",
        "label": LABEL_COL,
        "train_rows": train_df.count(),
        "test_rows": test_df.count(),
        "area_under_curve": float(binary_evaluator.evaluate(predictions)),
        "accuracy": float(accuracy_evaluator.evaluate(predictions)),
        "weighted_precision": float(precision_evaluator.evaluate(predictions)),
        "lgd_method": "Simple LTV-based LGD proxy; UPB is used as EAD"}

    confusion_matrix = predictions.groupBy(LABEL_COL, "prediction").count().orderBy(LABEL_COL, "prediction")

    os.makedirs("outputs/ML", exist_ok=True)
    with open("outputs/ML/model_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    predictions_df = caculate_credit_risk(predictions)

    predictions_df = predictions_df.select("loan_id", "reporting_month", LABEL_COL, 
                                            "prediction", "pd", "pd_band", "expected_loss")


    predictions_df.limit(10000).coalesce(1).write.mode("overwrite").option("header", True).csv("outputs/ML/predictions_sample")
    confusion_matrix.coalesce(1).write.mode("overwrite").option("header", True).csv("outputs/ML/confusion_matrix")

    model.write().overwrite().save("outputs/models/pd_logistic_regression")

    print("Model metrics:")
    print(json.dumps(metrics, indent=2))
    print("Confusion matrix:")
    confusion_matrix.show()

    return model, metrics


def caculate_credit_risk(df):
    """
    QUESTION ANSWERED:
    What is the expected loss for each loan record 

    Formula used:
    Expected Loss = PD * LGD * EAD
    """

    ead_df = df.withColumn("ead", coalesce(col("current_actual_upb"), 
                                           col("original_upb"), lit(0.0)))

    lgd_df = ead_df.withColumn("lgd", when(col("original_ltv").isNull(), lit(0.35))\
                               .when(col("original_ltv") <= 60, lit(0.20))
                               .when(col("original_ltv") <= 80, lit(0.30))
                               .when(col("original_ltv") <= 90, lit(0.40))
                               .when(col("original_ltv") <= 97, lit(0.50))
                               .otherwise(lit(0.60)))
    
    # The ML model returns a probability vector: [probability of 0, probability of 1].
    # Index 1 is the predicted probability of serious delinquency/default risk.
    pd_df = lgd_df.withColumn("pd", vector_to_array(col("probability"))[1])\
              .withColumn("pd_band", when(col("pd") < 0.01, "A: <1%")
                                    .when(col("pd") < 0.03, "B: 1%-3%")
                                    .when(col("pd") < 0.07, "C: 3%-7%")
                                    .when(col("pd") < 0.15, "D: 7%-15%")
                                    .otherwise("E: 15%+"))


    pd_df = pd_df.withColumn("expected_loss", col("pd") * col("lgd") * col("ead"))
    
    return pd_df


def build_transition_matrix(df):
    """
    QUESTION ANSWERED:
    How do loans migrate from one risk state to another month over month?

    Example: Current -> 30 DPD, 30 DPD -> Current, 60 DPD -> 90+ DPD.
    This is the transition matrix portion of the project.
    """
    window_spec = Window.partitionBy("loan_id").orderBy("reporting_date")

    transition_df = df.filter(col("reporting_date").isNotNull())\
                         .withColumn("next_reporting_month", lead("reporting_month").over(window_spec))\
                         .withColumn("next_risk_state", lead("risk_state").over(window_spec))\
                         .filter(col("next_risk_state").isNotNull())\
                         .select("loan_id", 
                                 col("reporting_month").alias("from_month"),
                                 col("next_reporting_month").alias("to_month"),
                                 col("risk_state").alias("from_state"),
                                 col("next_risk_state").alias("to_state"))

    state_counts_df = transition_df.groupBy("from_state", "to_state")\
        .agg(count("*").alias("transition_count"))

    from_totals_df = state_counts_df.groupBy("from_state").agg(
        spark_sum("transition_count").alias("from_state_total"))

    transition_matrix = state_counts_df\
        .join(from_totals_df, on="from_state", how="inner")\
        .withColumn("transition_probability", col("transition_count") / col("from_state_total"))\
        .orderBy("from_state", "to_state")

    transition_matrix_pivot = transition_matrix.groupBy("from_state").pivot("to_state")\
        .agg(avg("transition_probability")).orderBy("from_state")
    
    transition_df.coalesce(1).write.mode("overwrite").option("header", True).csv("outputs/TM/transition_DF")
    transition_matrix.coalesce(1).write.mode("overwrite").option("header", True).csv("outputs/TM/transition_matrix_long")
    transition_matrix_pivot.coalesce(1).write.mode("overwrite").option("header", True).csv("outputs/TM/transition_matrix_pivot")


    print("Transition matrix:")
    transition_matrix.show(truncate=False)
    return transition_matrix


def main() -> None:

    build_transition_matrix(modeling_df)
    train_pd_model(modeling_df)

    loaded_model = PipelineModel.load("outputs/models/pd_logistic_regression")

    # Get the logistic regression stage (it's the last stage)
    lr_model = loaded_model.stages[-1]

    # Print the coefficients
    print("Intercept:", lr_model.intercept)
    print("Coefficients:", lr_model.coefficients)

    print("Modeling and scenario analysis complete.")
    spark.stop()


if __name__ == "__main__":
    main()




