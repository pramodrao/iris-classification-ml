package com.pramodrao

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.ml.classification.RandomForestClassifier


object Main {

  val dataFile:String = "src/main/resources/iris.data"

  val schema = StructType (
      StructField("sepalLengthInCms", DoubleType, true) ::
      StructField("sepalWidthInCms", DoubleType, true) ::
      StructField("petalLengthInCms", DoubleType, true) ::
      StructField("petalWidthInCms", DoubleType, true) ::
      StructField("irisFlowerType", StringType, true) :: Nil
    )

  def main(args: Array[String]): Unit = {

    lazy val sparkSession = SparkSession.builder()
        .appName("Iris ML Example")
        .config("spark.master", "local").getOrCreate()

    val input = sparkSession.sparkContext.textFile(dataFile)
                    .flatMap(partition => partition.split("\n").toList)
                    .map(_.split(","))
                    .map(row => Row(row(0).toDouble, row(1).toDouble, row(2).toDouble, row(3).toDouble,row(4)))

    val rawData = sparkSession.createDataFrame(input, schema)

    val irisFeatureColumns = Array("sepalLengthInCms", "sepalWidthInCms", "petalLengthInCms", "petalWidthInCms")
    val assembler = new VectorAssembler().setInputCols(irisFeatureColumns).setOutputCol("features")
    val featureSet = assembler.transform(rawData)
    val labelIndexer = new StringIndexer().setInputCol("irisFlowerType").setOutputCol("label")
    val labeledSet = labelIndexer.fit(featureSet).transform(featureSet)

    val Array(trainingSet, testSet) = labeledSet.randomSplit(Array(0.70, 0.30), 25L)

    val evaluator:MulticlassClassificationEvaluator =
      new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")

    // Using Logistic Regression
    val lr = new LogisticRegression().setMaxIter(20).setRegParam(0.3).setFamily("multinomial")
    val model = lr.fit(trainingSet)
    val predictions = model.transform(testSet)
    println("Accuracy = " +evaluator.evaluate(predictions))
    predictions.select ("features", "label", "prediction").show()

    // Using Random Forest
    val randomForestClassifier = new RandomForestClassifier().setSeed(5043L)
    val modelRF = randomForestClassifier.fit(trainingSet)
    val predictionsRF = modelRF.transform(testSet)

    println("Accuracy = " +evaluator.evaluate(predictionsRF))
    predictionsRF.select ("features", "label", "prediction").show()
  }
}