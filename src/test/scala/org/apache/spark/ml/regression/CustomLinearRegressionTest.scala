package org.apache.spark.ml.regression

import org.apache.spark.ml.linalg.{Matrices, Vector, Vectors}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, udf}
import org.scalatest.flatspec._
import org.scalatest.matchers._

import scala.util.Random

class CustomLinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

  val delta = 0.00001

  lazy val data: DataFrame = CustomLinearRegressionTest._data
  lazy val vectors: Seq[Vector] = CustomLinearRegressionTest._vectors

  val predictUDF: UserDefinedFunction = CustomLinearRegressionTest._predictUDF

  "Model" should "predict input data" in {
    val model: CustomLinearRegressionModel = new CustomLinearRegressionModel(w = Vectors.dense(1.5, 0.3, -0.7))
    validateModel(model.transform(data))
  }

  "Estimator" should "produce functional model" in {
    val estimator = new CustomLinearRegression()

    val dataset = data.withColumn("label", predictUDF(col("features")))
    val model = estimator.fit(dataset)

    validateModel(model.transform(dataset))
  }

  "Estimator" should "predict correctly" in {
    val estimator = new CustomLinearRegression().setMaxIter(8192)

    import sqlContext.implicits._

    val randomData = Matrices
      .rand(30000, 3, Random.self)
      .rowIter
      .toSeq
      .map(x => Tuple1(x))
      .toDF("features")

    val dataset = randomData.withColumn("label", predictUDF(col("features")))
    val model = estimator.fit(dataset)

    model.w(0) should be(1.5 +- delta)
    model.w(1) should be(0.3 +- delta)
    model.w(2) should be(-0.7 +- delta)
  }

  private def validateModel(data: DataFrame): Unit = {
    val vector = data.collect().map(_.getAs[Double](1))

    vector.length should be(2)

    vector(0) should be(21.05 +- delta)
    vector(1) should be(-1.6 +- delta)
  }

}

object CustomLinearRegressionTest extends WithSpark {

  lazy val _vectors: Seq[Vector] = Seq(
    Vectors.dense(13.5, 12, 4),
    Vectors.dense(-1, 2, 1)
  )

  lazy val _data: DataFrame = {
    import sqlContext.implicits._
    _vectors.map(x => Tuple1(x)).toDF("features")
  }

  val _predictUDF: UserDefinedFunction = udf { features: Any =>
    val arr = features.asInstanceOf[Vector].toArray

    1.5 * arr.apply(0) + 0.3 * arr.apply(1) - 0.7 * arr.apply(2)
  }

}