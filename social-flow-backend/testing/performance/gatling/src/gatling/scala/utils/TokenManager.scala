package utils

import io.gatling.core.Predef._
import scala.concurrent.duration._
import io.gatling.http.Predef._
import io.gatling.core.session.Session

/**
 * TokenManager provides helper chains to acquire and inject tokens into session.
 *
 * Two strategies:
 *  - lightweight per-user login via /auth/login
 *  - cached token in system under test (not possible from Gatling client side)
 *
 * This object exposes a function to call in scenarios to obtain a token and store in session
 * as "authToken".
 */
object TokenManager {
  def loginFlow(usernameSessionKey: String = "username", passwordSessionKey: String = "password"): ChainBuilder = {
    exec(http("Auth - login")
      .post("/auth/login")
      .body(StringBody(session => {
        val user = session(usernameSessionKey).as[String]
        val pass = session(passwordSessionKey).as[String]
        s"""{"username":"$user","password":"$pass"}"""
      }))
      .check(status.is(200))
      .check(jsonPath("$.token").saveAs("authToken"))
      .check(jsonPath("$.expires_in").optional.saveAs("tokenExpires"))
    )
  }

  // helper to add Authorization header when making a request
  def withAuth(chain: ChainBuilder): ChainBuilder = {
    exec(session => {
      // add header at request time; usage: exec(session => session.set("authHeader", s"Bearer ${session("authToken").as[String]}"))
      session
    }).doIf(session => session.contains("authToken")) {
      chain
    }
  }
}
