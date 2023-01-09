package org.apache.spark.ml.homework2

import org.scalatest._
import org.scalatest.flatspec._
import org.scalatest.matchers._


class StartSparkTest extends AnyFlatSpec with should.Matchers with WithSpark {

  "Spark" should "start context" in {
    val s = spark

    Thread.sleep(60000)
  }

}
