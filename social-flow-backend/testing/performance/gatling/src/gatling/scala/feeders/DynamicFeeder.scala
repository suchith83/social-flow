package feeders

import io.gatling.core.feeder.Feeder
import io.gatling.core.Predef._
import java.time.Instant

/**
 * Dynamic feeder that can be used to generate unique keys/payloads
 * per virtual user iteration. Useful for creating unique upload keys
 * or item names to avoid collisions.
 */
object DynamicFeeder {
  def uniqueKeyFeeder(prefix: String = "perf"): Feeder[String] = {
    Iterator.continually(Map("uniqueKey" -> s"${prefix}-${Instant.now().toEpochMilli}-${scala.util.Random.nextInt(9999)}"))
  }

  def timestampFeeder(): Feeder[String] = {
    Iterator.continually(Map("ts" -> Instant.now.getEpochSecond.toString))
  }
}
