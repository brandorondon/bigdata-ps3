import org.apache.hadoop.io.Text
import org.apache.hadoop.io.LongWritable
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating
import com.github.fommil.netlib._

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
val products = reviewMap.map(_._2).distinct.sortBy(x => x).zipWithIndex.collectAsMap
val ratings = reviewMap.collect.map( r => new Rating(names(r._1).toInt, products(r._2)toInt, r._3.toDouble))
val ratingsRdd = sc.parallelize(ratings)
val rank = 10
val numIterations = 10
val model = ALS.train(ratingsRdd, rank, numIterations, 0.01)