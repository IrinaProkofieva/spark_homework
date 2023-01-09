package org.apache.spark.ml.homework2

import breeze.linalg.{DenseMatrix, DenseVector, InjectNumericOps}
import breeze.numerics.exp
import breeze.stats.mean
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.linalg.{Matrix, VectorUDT, Vectors}
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.param.{DoubleParam, IntParam, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable, SchemaUtils}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}

trait LinearRegressionParams extends HasInputCol with HasOutputCol {
  def setInputCol(value: String): this.type  = set(inputCol, value)
  def setOutputCol(value: String): this.type  = set(outputCol, value)

  val maxIter = new IntParam(this, "maxIter", "Maximum number of iteration")
  val lr = new DoubleParam(this, "lr", "Learning rate")
  val threshold = new DoubleParam(this, "threshold", "Threshold")

  def getMaxIter : Int = $(maxIter)
  def setMaxIter(value: Int) : this.type = set(maxIter, value)
  setDefault(maxIter -> 10)

  def getLR : Double = $(lr)
  def setLR(value: Double) : this.type = set(lr, value)
  setDefault(lr -> 0.001)

  def getThreshold : Double = $(threshold)
  def setThreshold(value: Double) : this.type = set(threshold, value)
  setDefault(threshold -> 0.5)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getInputCol, new VectorUDT())

    if (schema.fieldNames.contains($(outputCol))) {
      SchemaUtils.checkColumnType(schema, getOutputCol, new VectorUDT())
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getInputCol).copy(name = getOutputCol))
    }
  }
}

class LinearRegression(override val uid: String) extends Estimator[LinearRegressionModel]
  with LinearRegressionParams {

  def this() = this(Identifiable.randomUID("linearRegression"))

  var inputs: Dataset[Vector[Double]] = null
  var X: DenseMatrix[Double] = null
  var y: DenseVector[Double] = null
  var W: DenseVector[Double] = null
  var b: Double = 0

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {

    implicit val encoder : Encoder[Vector[Double]] = ExpressionEncoder()
    implicit val encoder2 : Encoder[Double] = ExpressionEncoder()

    inputs = dataset.select(dataset($(inputCol)).as[Vector[Double]])

    X = getDenseMatrixFromDF(inputs.toDF())

    var dim: Int = X.cols

    y = DenseVector(dataset.select(dataset($(outputCol)).as[Double]).collect())

    W = DenseVector.zeros(dim)

    b = 0

    var i: Int = 0

    while (i < getMaxIter) {
      update_weights()
      i = i + 1
    }

    copyValues(new LinearRegressionModel(W, b))
  }

  private def getDenseMatrixFromDF(featuresDF:DataFrame):DenseMatrix[Double] = {
    val featuresTrain = featuresDF.columns
    val rows = featuresDF.count().toInt

    val newFeatureArray:Array[Double] = featuresTrain
      .indices
      .flatMap(i => featuresDF
        .select(featuresTrain(i))
        .collect())
      .map(r => r.toSeq.toArray).toArray.flatten.flatMap(_.asInstanceOf[org.apache.spark.ml.linalg.DenseVector].values)

    val newCols = newFeatureArray.length / rows
    val denseMat:DenseMatrix[Double] = new DenseMatrix[Double](rows, newCols, newFeatureArray)
    denseMat
  }

  def update_weights(): Unit = {
    var preds: DenseVector[Double] = -(X * W) + b
    preds = preds.map(v => 1 / (1 + exp(v)))
    val difference: DenseVector[Double] = preds - y
//    val zeroLoss: DenseVector[Double] = y_true *:* lgrthm(preds +1e-9)
//    y_true.foreachValue(v => 1-v)
//    preds.foreachValue(v => 1-v)
//    val oneLoss: DenseVector[Double] = y_true *:* lgrthm(preds +1e-9)
//    val loss: Double = -mean(zeroLoss + oneLoss)
    val gradB: Double = mean(difference)
    var gradW: DenseVector[Double] = X.t * difference
    gradW = gradW.map(v => v / gradW.size)
    W = W - getLR * gradW
    b = b - getLR * gradB
  }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
}

object LinearRegression extends DefaultParamsReadable[LinearRegression]

class LinearRegressionModel private[homework2](
                                                override val uid: String,
                                                val weights: DenseVector[Double],
                                                val bias: Double
                                              ) extends Model[LinearRegressionModel]
                                              with LinearRegressionParams {
  private[homework2] def this(weights: DenseVector[Double], bias: Double) = this(
    Identifiable.randomUID("linearRegressionModel"),
    weights,
    bias
  )

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(
    new LinearRegressionModel(weights, bias), extra
  )

  override def transform(dataset: Dataset[_]): DataFrame = {
    val predict = {
      dataset.sqlContext.udf.register(uid+"_predict",
        (x : org.apache.spark.ml.linalg.Vector) => {
          val denseX: DenseVector[Double] = new DenseVector[Double](x.toArray)
          val preds: Double = 1 / (1 + exp(-denseX.dot(weights) + bias))
          if (preds > getThreshold) {
            1
          }
          else {
            0
          }
        }
      )
    }
    dataset.withColumn($(outputCol), predict(dataset($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
}