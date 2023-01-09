name := "itmo"

version := "0.1"

scalaVersion := "2.13.10"

val sparkVersion = "3.2.2"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-sql" % sparkVersion withSources(),
  "org.apache.spark" %% "spark-mllib" % sparkVersion withSources()
)

libraryDependencies += ("org.scalatest" %% "scalatest" % "3.2.14" % "test" withSources())