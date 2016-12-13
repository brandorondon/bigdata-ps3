// $SPARK_HOME/bin/spark-submit --master yarn --deploy-mode client --queue hadoop07 --driver-memory 4g --executor-memory 4g --executor-cores 2 recommender.jar

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
import java.io.BufferedOutputStream


object Main {
	def main(args: Array[String]) {
		val sc = new SparkContext

		val conf = new Configuration
		conf.set("textinputformat.record.delimiter", "\n\n")

		val rawData = sc.newAPIHadoopFile("/shared3/data-small.txt", classOf[TextInputFormat], classOf[LongWritable], classOf[Text], conf).map(_._2.toString)
		val splitReviews = rawData.map( review => review.split("\n") )
		val reviewMap = splitReviews.map( arr => {
		  val productId = arr( 0 ).split("/productId: ")(1)
		  val userId = arr( 3 ).split("/userId: ")(1)
		  val score = arr( 6 ).split("/score: ")(1)
		  (userId, productId, score)
		} )

		val names = reviewMap.map(_._1).distinct.sortBy(x => x).zipWithIndex.collectAsMap
		val reverseNames = names.map(_.swap)
		val products = reviewMap.map(_._2).distinct.sortBy(x => x).zipWithIndex.collectAsMap
		val reverseProducts = products.map(_.swap)
		val ratings = reviewMap.collect.map( r => new Rating(names(r._1).toInt, products(r._2)toInt, r._3.toDouble))
		val ratingsRdd = sc.parallelize(ratings)
		val rank = 10
		val numIterations = 10
		val model = ALS.train(ratingsRdd, rank, numIterations, 0.01)

		conf.set("textinputformat.record.delimiter", "\n")
		val ratingsToProcess = sc.textFile("/user/hadoop07/items-small.txt").collect
		// To test on the keySet which we know exists in dict
		//val ratingsToProcess = products.keySet.toArray.take(10)

		val finalResults = ratingsToProcess.map( curr => {
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

		val outputFileName = "/user/hadoop07/recommended-items.txt"
		val fs = FileSystem.get(sc.hadoopConfiguration);
		val output = fs.create(new Path(outputFileName));
		val os = new BufferedOutputStream(output)
		for (line <- finalResults) {
			os.write((line + "\n").getBytes("UTF-8"))
		}

		os.close()
	}
}
