package simulations

import io.gatling.core.Predef._
import io.gatling.http.Predef._
import scala.concurrent.duration._
import utils.MultipartUploadHelper
import feeders.DynamicFeeder

/**
 * Storage upload/download simulation:
 * - For upload: request presigned-form then perform multipart upload
 * - For download: hit /storage/presigned?key=... and then GET that URL
 *
 * Note: the actual multipart upload logic depends on your app's presigned contract.
 */
class StorageSimulation extends Simulation {
  val httpProtocol = http
    .baseUrl(sys.env.getOrElse("BASE_URL", "http://localhost:3000/api"))
    .acceptHeader("application/json")

  val feeder = DynamicFeeder.uniqueKeyFeeder("up")

  val uploadScenario = scenario("Storage Upload")
    .feed(feeder)
    .exec(MultipartUploadHelper.preparePresignedForm("uploadKey"))
    .pause(200.millis)
    .exec(MultipartUploadHelper.uploadFileFromResource("src/gatling/resources/data/sample-files/sample.txt"))

  val downloadScenario = scenario("Storage Download")
    .exec(session => session.set("downloadKey", "uploads/sample.txt"))
    .exec(http("Get presigned download")
      .get("/storage/presigned")
      .queryParam("key", "${downloadKey}")
      .check(status.is(200))
      .check(jsonPath("$.url").saveAs("downloadUrl"))
    )
    .pause(200.millis)
    .exec(http("Fetch presigned asset")
      .get("${downloadUrl}")
      .check(status.in(200))
    )

  setUp(
    uploadScenario.inject(rampUsersPerSec(1) to 10 during 30.seconds),
    downloadScenario.inject(constantUsersPerSec(5) during 60.seconds)
  ).protocols(httpProtocol)
    .assertions(global.successfulRequests.percent.gte(98))
}
