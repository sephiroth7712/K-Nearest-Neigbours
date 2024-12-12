import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import breeze.linalg.normalize
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.Row
import scala.collection.mutable.ArrayBuffer
import breeze.linalg.DenseVector
import org.apache.spark.ml.linalg
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.Dataset

object KNearestNeighbours {
  def main(args: Array[String]): Unit = {
    val dataset = "large"
    val k = 3
    val spark = SparkSession.builder
      .appName("KNearestNeighbours")
      .config("spark.master", "local[*]")
      .getOrCreate()
    import spark.implicits._
    spark.sparkContext.setLogLevel("ALL")

    val startTime = System.currentTimeMillis()
    val csvTrain = spark.read
      .format("csv")
      .option("header", false)
      .option("inferSchema", true)
      .load("datasets/" + dataset + "-train.csv")
      .repartition(10) // Comment this line for small dataset

    val numFeatures = csvTrain.columns.length

    val csvTest = spark.read
      .format("csv")
      .option("header", false)
      .option("inferSchema", true)
      .load("datasets/" + dataset + "-test.csv")

    val assembler = new VectorAssembler()
      .setInputCols(csvTest.columns.dropRight(1))
      .setOutputCol("features")

    val train =
      assembler
        .transform(csvTrain)
        .drop(csvTrain.columns.dropRight(1): _*)
        .withColumnRenamed(csvTrain.columns.last, "class")
        .cache()
    val test =
      assembler
        .transform(csvTest)
        .drop(csvTest.columns.dropRight(1): _*)
        .withColumnRenamed(csvTest.columns.last, "class")
        .withColumn("id", monotonically_increasing_id())
        .cache()

    val testCount = test.count().intValue()
    def euclideanDistance(v1: linalg.Vector, v2: linalg.Vector): Double = {
      math.sqrt(Vectors.sqdist(v1, v2))
    }

    def kNN(
        iter: Iterator[Row],
        testData: Array[Row]
    ): Iterator[(Int, Array[(Int, Double)])] = {

      val CD = new Array[(Int, Array[(Int, Double)])](testCount)

      val tr: ArrayBuffer[linalg.DenseVector] = new ArrayBuffer
      val classes: ArrayBuffer[Int] = new ArrayBuffer

      while (iter.hasNext) {
        val row = iter.next()
        val features = row.getAs[linalg.DenseVector]("features")
        tr.append(features)
        val label = row.getAs[Int]("class")
        classes.append(label)
      }

      testData.foreach(row => {
        val id = row.getAs[Long]("id").toInt
        val CDInstance: Array[(Int, Double)] =
          Array.fill[(Int, Double)](k)(-1, Double.MaxValue)
        for (i <- 0 until tr.length) {
          val dist =
            euclideanDistance(row.getAs[linalg.DenseVector]("features"), tr(i))

          var breakLoop = false
          for (j <- 0 until k) {
            if (dist < CDInstance(j)._2 && !breakLoop) {
              for (x <- k - 2 to j by -1) {
                CDInstance(x + 1) = CDInstance(x)
              }
              CDInstance(j) = (classes(i), dist)
              breakLoop = true
            }
          }
        }
        CD(id) = (id, CDInstance)
      })

      CD.iterator
    }

    val testArray = test.collect()
    val resultKNN = train
      .mapPartitions(iter => kNN(iter, testArray))

    def getMinK(
        cdkv1: (Int, Array[(Int, Double)]),
        cdkv2: (Int, Array[(Int, Double)])
    ): (Int, Array[(Int, Double)]) = {
      var cont = 0
      val cd1 = cdkv1._2
      val cd2 = cdkv2._2
      for (i <- 0 until k) {
        if (cd1(cont)._2 < cd2(i)._2) {
          cd2(i) = cd1(cont)
          cont += 1
        }
      }
      (cdkv1._1, cd2)
    }

    val result = resultKNN
      .map(row => (row._1, row._2))
      .groupByKey(_._1)
      .reduceGroups((cdkv1, cdkv2) => getMinK(cdkv1, cdkv2))
      .map(_._2)

    val getClassUDF = udf((array: Seq[(Int, Double)]) => {
      array.groupBy(_._1).maxBy(_._2.length)._1
    })

    val resultDF = result
      .withColumn("exploded", explode($"_2"))
      .groupBy($"_1")
      .agg(
        getClassUDF(collect_list($"exploded").as("classes"))
          .as("class")
      )

    resultDF.write.csv(dataset + "-output")
    val end_time = System.currentTimeMillis() - startTime
    println(s"Program took $end_time ms")
    spark.stop()
  }
}
