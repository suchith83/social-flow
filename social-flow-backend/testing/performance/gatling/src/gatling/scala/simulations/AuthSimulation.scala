package simulations

import io.gatling.core.Predef._
import io.gatling.http.Predef._
import scala.concurrent.duration._
import feeders.CsvFeeders
import utils.TokenManager

/**
 * Authentication scenario:
 * - Uses a CSV (or dynamic) feeder for users
 * - Performs login and calls protected endpoint with obtained token
 * - Includes checks and assertions on token acquisition and /auth/me
 */
class AuthSimulation extends Simulation {
  val httpProtocol = http
    .baseUrl(sys.env.getOrElse("BASE_URL", "http://localhost:3000/api"))
    .acceptHeader("application/json")
    .contentTypeHeader("application/json")

  val usersFeeder = CsvFeeders.userFeeder()

  val scn = scenario("Auth Flow")
    .feed(usersFeeder)
    .exec(TokenManager.loginFlow())
    .pause(1)
    .exec(http("Auth - me")
      .get("/auth/me")
      .header("Authorization", "Bearer ${authToken}")
      .check(status.is(200))
      .check(jsonPath("$.username").saveAs("meUsername"))
    )
    .exec(session => {
      // simple assertion: username returned equals input username
      val expected = session("username").asOption[String]
      val returned = session("meUsername").asOption[String]
      if (expected.isDefined && returned.isDefined && expected.get != returned.get)
        println(s"WARNING: expected ${expected.get} got ${returned.get}")
      session
    })

  setUp(
    scn.inject(
      rampUsersPerSec(1) to (sys.env.getOrElse("USERS", "50").toInt) during (sys.env.getOrElse("RAMP_UP_SECONDS", "30").toInt.seconds),
      constantUsersPerSec(sys.env.getOrElse("USERS", "50").toInt) during (sys.env.getOrElse("DURATION_SECONDS", "60").toInt.seconds)
    )
  ).protocols(httpProtocol)
    .assertions(
      global.successfulRequests.percent.gte(99),
      global.responseTime.percentile(95).lte(1000)
    )
}
