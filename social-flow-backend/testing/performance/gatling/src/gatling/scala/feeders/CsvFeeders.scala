package feeders

import io.gatling.core.feeder.Feeder
import scala.io.Source
import scala.util.Random

/**
 * Simple CSV feeder loader helper.
 * Places file under resources/data/users.csv
 */
object CsvFeeders {
  def userFeeder(resourcePath: String = "data/users.csv"): Feeder[String] = {
    val src = Source.fromResource(resourcePath)
    val lines = src.getLines().toVector
    src.close()
    // Expect header: username,password
    val header :: rest = lines.toList
    val cols = header.split(",").map(_.trim)
    val rows = rest.map { line =>
      val parts = line.split(",").map(_.trim)
      Map(cols.zip(parts): _*)
    }
    io.gatling.core.feeder.IteratorFeederBuilder(rows.toIterator)
  }

  // convenience generator for synthetic user names
  def randomUserFeeder(): Feeder[String] = {
    Iterator.continually(Map("username" -> s"testuser_${Random.alphanumeric.take(8).mkString}", "password" -> "P@ssw0rd"))
  }
}
