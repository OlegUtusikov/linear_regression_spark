package org.apache.spark.ml.regression

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.shared.{HasMaxIter, HasTol}
import org.apache.spark.ml.param.{DoubleParam, Param, ParamMap}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model, PredictorParams}
import org.apache.spark.mllib.linalg.{Vectors => MLLibVectors}
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}
import breeze.linalg.{DenseVector, max}
import breeze.numerics.abs

trait CustomLinearRegressionParams extends PredictorParams with HasMaxIter with HasTol {

  def setMaxIter(value: Int): this.type = set(maxIter, value)

  final val gradRate: Param[Double] = new DoubleParam(
    this,
    "gradientRate",
    "Gradient descent rate"
  )

  def setGradRate(value: Double): this.type = set(gradRate, value)

  def getGradRate: Double = $(gradRate)

  setDefault(maxIter -> 2048, tol -> 1e-8, gradRate -> 4e-2)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getFeaturesCol, new VectorUDT())

    if (schema.fieldNames.contains($(predictionCol))) {
      SchemaUtils.checkColumnType(schema, getPredictionCol, new VectorUDT())
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getFeaturesCol).copy(name = getPredictionCol))
    }
  }
}

class CustomLinearRegression(override val uid: String) extends Estimator[CustomLinearRegressionModel] with CustomLinearRegressionParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("linearRegression"))

  override def fit(dataset: Dataset[_]): CustomLinearRegressionModel = {
    implicit val vectorEncoder: Encoder[Vector] = ExpressionEncoder()
    val asm = new VectorAssembler().setInputCols(Array(getFeaturesCol, getLabelCol)).setOutputCol("result")
    val vectors = asm.transform(dataset).select("result").as[Vector]

    val cnt = vectors.first().size - 1
    val max_iter = getMaxIter
    val tol = getTol
    val gr = getGradRate

    val w = DenseVector.fill(cnt, 0d)
    var diff_w = w - DenseVector.fill(cnt, Double.PositiveInfinity)
    var cur_iter = 0
    while (cur_iter < max_iter && max(abs(diff_w)) > tol) {
      val grads = vectors.rdd.mapPartitions(part => {
        val summarizer = new MultivariateOnlineSummarizer()
        part.foreach(row => {
          val x_ = row.asBreeze(0 until cnt).toDenseVector
          val y = row.asBreeze(cnt)
          summarizer.add(MLLibVectors.fromBreeze((x_.dot(w) - y) * x_))
        })
        Iterator(summarizer)
      }).reduce(_ merge _)

      diff_w = w.copy
      w -= 2 * gr * grads.mean.asBreeze
      diff_w -= w
      cur_iter += 1
    }

    copyValues(new CustomLinearRegressionModel(Vectors.fromBreeze(w)).setParent(this))
  }

  override def copy(extra: ParamMap): Estimator[CustomLinearRegressionModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
}

class CustomLinearRegressionModel(override val uid: String,
                                  val w: Vector
                                 ) extends Model[CustomLinearRegressionModel] with CustomLinearRegressionParams {

  private[regression] def this(w: Vector) = this(
    Identifiable.randomUID("linearRegressionModel"),
    w.toDense
  )

  override def transformSchema(schema: StructType): StructType = {
    var outputSchema = validateAndTransformSchema(schema)
    if ($(predictionCol).nonEmpty) {
      outputSchema = SchemaUtils.updateNumeric(outputSchema, $(predictionCol))
    }
    outputSchema
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    val outputSchema = transformSchema(dataset.schema, logging = false)
    val predictUDF = udf { features: Any =>
      predict(features.asInstanceOf[Vector])
    }

    dataset.withColumn($(predictionCol), predictUDF(col($(featuresCol))), outputSchema($(predictionCol)).metadata)
  }

  override def copy(extra: ParamMap): CustomLinearRegressionModel = copyValues(new CustomLinearRegressionModel(w), extra)

  private def predict(features: Vector) = features.asBreeze.dot(w.asBreeze)

}
