package simulations

import io.gatling.core.Predef._
import io.gatling.http.Predef._
import scala.concurrent.duration._
import feeders.DynamicFeeder

/**
 * Simulate primary down and test failover behavior:
 * - Calls a test-only endpoint to mark primary down (simulate)
 * - Performs operations (uploads) and checks status
 * - Restores primary
 *
 * This requires your system to provide testing toggles (e.g., /testing/toggle-primary?state=down)
 * If not available, adapt to your environment (e.g., flip feature flag via API).
 */
class FailoverSimulation extends Simulation {
  val httpProtocol = http
    .baseUrl(sys.env.getOrElse("BASE_URL", "http://localhost:3000/api"))
    .acceptHeader("application/json")

  val feeder = DynamicFeeder.uniqueKeyFeeder("fail")

  val scn = scenario("Multi-cloud failover")
    .feed(feeder)
    .exec(http("Get current provider").get("/multi-cloud/provider").check(status.is(200)))
    .pause(200.millis)
    .exec(http("Simulate primary down")
      .post("/testing/toggle-primary")
      .body(StringBody("""{ "state": "down" }""")).asJson
      .check(status.in(200,202))
    )
    .pause(1)
    .exec(http("Perform upload operation (failover expected)")
      .post("/multi-cloud/operate")
      .body(StringBody(session => s"""{ "action":"upload", "key":"${session("uniqueKey").as[String]}" }""")).asJson
      .check(status.is(200))
      .check(jsonPath("$.status").saveAs("opStatus"))
    )
    .exec(session => {
      val s = session("opStatus").asOption[String]
      if (s.isDefined && s.get != "ok") println(s"operation status: ${s.get}")
      session
    })
    .pause(200.millis)
    .exec(http("Restore primary")
      .post("/testing/toggle-primary")
      .body(StringBody("""{ "state": "up" }""")).asJson
      .check(status.in(200,202))
    )

  setUp(
    scn.inject(
      rampUsersPerSec(1) to 5 during 30.seconds,
      constantUsersPerSec(3) during 120.seconds
    )
  ).protocols(httpProtocol)
    .assertions(
      global.responseTime.percentile(95).lte(2500),
      global.successfulRequests.percent.gte(95)
    )
}
