// --- USAGE ---
// $SPARK_HOME/bin/spark-submit --master yarn --deploy-mode client --queue hadoop07 --driver-memory 4g --executor-memory 4g --executor-cores 2 recommender_2.11-1.0.jar
// -------------

import org.apache.hadoop.io.Text
import org.apache.hadoop.fs.FileSystem
import org.apache.hadoop.io.LongWritable
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import scala.collection.{Map, mutable}
import java.io.BufferedOutputStream


object Main {
	// Write output file to HDFS
	def writeResultToHDFS(fname: String, result: Array[String], sc: SparkContext) = {
		val fs = FileSystem.get(sc.hadoopConfiguration)
		val output = fs.create(new Path(fname))
		val os = new BufferedOutputStream(output)
		for (line <- result) {
			os.write((line + "\n").getBytes("UTF-8"))
		}

		os.close()
	}

  // Read in HDFS amazon reviews and return reference to the RDD
	def preProcessData(fname: String, sc: SparkContext): RDD[(String, String, String)] =  {
		val conf = new Configuration
		conf.set("textinputformat.record.delimiter", "\n\n")

		val rawData = sc.newAPIHadoopFile(fname, classOf[TextInputFormat], classOf[LongWritable], classOf[Text], conf).map(_._2.toString)
		val splitReviews = rawData.map( review => review.split("\n") )
		val reviewMap = splitReviews.map( arr => {
			val productId = arr( 0 ).split("/productId: ")(1)
			val userId = arr( 3 ).split("/userId: ")(1)
			val score = arr( 6 ).split("/score: ")(1)
			(userId, productId, score)
		} )
		return reviewMap
	}

	// Create model using ALS matrix factorization algorithm
	def createModelWithALS(reviewMap: RDD[(String, String, String)], names: Map[String, Long], products: Map[String, Long], sc: SparkContext): MatrixFactorizationModel = {
		val ratings = reviewMap.collect.map( r => new Rating(names(r._1).toInt, products(r._2)toInt, r._3.toDouble))
		val ratingsRdd = sc.parallelize(ratings)
		val rank = 10
		val numIterations = 10
		return ALS.train(ratingsRdd, rank, numIterations, 0.01)
	}

	// Get 10 recommendations for each item in the item input list
	def getRecommendedItems(itemListFName: String, model: MatrixFactorizationModel, names: Map[String, Long], products: Map[String, Long], sc: SparkContext): Array[String] = {
		val conf = new Configuration
		val reverseNames = names.map(_.swap)
		val reverseProducts = products.map(_.swap)
		conf.set("textinputformat.record.delimiter", "\n")
		val ratingsToProcess = sc.textFile(itemListFName).collect
		val recommendedItemList = ratingsToProcess.map( curr => {
			if ( products.contains(curr)) {
				val tempCurr = products(curr)
				val numUsers = 100
				val currRecommended = model.recommendUsers(tempCurr.toInt, numUsers)
				.map( r => model.recommendProducts(r.user,1))
				.map( r => r(0).product)
				.filter(_ != tempCurr)
				.take(10)
				.map( r => reverseProducts(r))
				curr + "," + currRecommended.mkString(",")
			} else {
				curr
			}
		})
		return recommendedItemList
	}

	def main(args: Array[String]) {
		val sc = new SparkContext

		val reviewMap = preProcessData("/shared3/data-medium.txt", sc)
		// Mapping of username to integer
		val nameMap = reviewMap.map(_._1).distinct.sortBy(x => x).zipWithIndex.collectAsMap
		// Mapping of product_id to integer
		val productMap = reviewMap.map(_._2).distinct.sortBy(x => x).zipWithIndex.collectAsMap
		val model = createModelWithALS(reviewMap, nameMap, productMap, sc)
		val finalResults = getRecommendedItems("/shared3/items-medium.txt", model, nameMap, productMap, sc)

		writeResultToHDFS("/user/hadoop07/recommended-items.txt", finalResults, sc)
	}
}
