package org.apache.spark.ml.homework2

import breeze.linalg.DenseVector
import org.apache.spark.ml.homework2.LinearRegressionTest.{_test_data, _train_data}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.DataFrame
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should

class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

  val lr = 0.1
  val delta = 0.0000001
  lazy val test_data: DataFrame = LinearRegressionTest._test_data
  lazy val train_data: DataFrame = LinearRegressionTest._train_data


  "Model" should "predict" in {
    val model: LinearRegressionModel = new LinearRegressionModel(
      weights = new DenseVector(Array(1, 2.3, 4, -9)),
      bias = 1.5
    ).setInputCol("features")
      .setOutputCol("predictions")

    val preds: Array[Int] = model.transform(test_data).collect().map(_.getAs[Int](1))
    preds should be(Array(1, 1, 0))
  }

  "Estimator" should "train" in {
    val estimator = new LinearRegression()
      .setInputCol("features")
      .setOutputCol("label")

    val model = estimator.fit(train_data)

    //data from python analog
    model.bias should be(-0.0014892 +- delta)
    model.weights(0) should be(-0.0141814 +- delta)
    model.weights(1) should be(0.0231037 +- delta)
    model.weights(2) should be(0.0091981 +- delta)
    model.weights(3) should be(0.0133102 +- delta)
  }

  "Estimator" should "produce good model" in {
    val estimator = new LinearRegression()
      .setInputCol("features")
      .setOutputCol("label")
      .setLR(0.0001)
      .setMaxIter(100)

    val model = estimator.fit(train_data)
    val preds: Array[Int] = model.transform(test_data).collect().map(_.getAs[Int](1))
    preds should be(Array(0, 0, 1))
  }

}

object LinearRegressionTest extends WithSpark {

//  lazy val _test_data = spark.read.csv("src/test/scala/org/apache/spark/ml/homework2/test_data.csv")

  lazy val _test_vectors = Seq(
    Vectors.dense(9, 7.5, 4, -20),
    Vectors.dense(-1, 0, 2, -4),
    Vectors.dense(5.5, 3, 2.5, 16.5)
  )

  lazy val _test_data: DataFrame = {
    import sqlc.implicits._
    _test_vectors.map(x => Tuple1(x)).toDF("features")
  }

  lazy val _train_vectors = Seq(
    (Vectors.dense(9, 7.5, 4, -20), 0),
    (Vectors.dense(-1, 0, 2, -4), 0),
    (Vectors.dense(5.5, 3, 2.5, 16.5), 1)
  )

  lazy val _train_data: DataFrame = {
    import sqlc.implicits._
    _train_vectors.toDF("features", "label")
  }
}