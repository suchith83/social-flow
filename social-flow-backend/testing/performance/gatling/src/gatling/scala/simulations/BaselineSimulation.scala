package simulations

import io.gatling.core.Predef._
import io.gatling.http.Predef._
import scala.concurrent.duration._
import io.gatling.core.structure.ScenarioBuilder
import feeders.DynamicFeeder

/**
 * Baseline mixed scenario: health, read, write, list
 * - Demonstrates POST create then GET read using returned id (or random id)
 */
class BaselineSimulation extends Simulation {
  val httpProtocol = http
    .baseUrl(sys.env.getOrElse("BASE_URL", "http://localhost:3000/api"))
    .acceptHeader("application/json")
    .contentTypeHeader("application/json")

  val feeder = DynamicFeeder.uniqueKeyFeeder("item")

  val scn: ScenarioBuilder = scenario("Baseline API Scenario")
    .feed(feeder)
    .exec(http("Health check").get("/health").check(status.is(200)))
    .pause(100.millis)
    .exec(http("Create item")
      .post("/items")
      .body(StringBody(session => {
        val name = session("uniqueKey").as[String]
        s"""{"name":"$name","description":"load test item"}"""
      }))
      .check(status.in(200, 201))
      .check(jsonPath("$.id").saveAs("createdId"))
    )
    .pause(100.millis)
    .doIf(session => session.contains("createdId")) {
      exec(http("Get created item")
        .get("/items/${createdId}")
        .check(status.in(200,404))
      )
    }
    .exec(http("List items")
      .get("/items?limit=20")
      .check(status.is(200))
    )

  setUp(
    scn.inject(
      rampUsersPerSec(5) to (sys.env.getOrElse("USERS","100").toInt) during (sys.env.getOrElse("RAMP_UP_SECONDS","30").toInt.seconds),
      constantUsersPerSec(sys.env.getOrElse("USERS","100").toInt) during (sys.env.getOrElse("DURATION_SECONDS","60").toInt.seconds)
    )
  ).protocols(httpProtocol)
    .assertions(
      forAll.responseTime.percentile(95).lte(1200),
      global.failedRequests.count.lte(10)
    )
}
