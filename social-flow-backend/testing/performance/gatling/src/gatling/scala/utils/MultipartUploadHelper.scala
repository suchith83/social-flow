package utils

import io.gatling.core.Predef._
import io.gatling.http.Predef._
import scala.concurrent.duration._
import java.nio.file.Files
import java.nio.file.Paths
import scala.util.Try

/**
 * Multipart upload helper demonstrates requesting a presigned upload form and then
 * performing a direct multipart/form-data POST to the storage host.
 *
 * Note: Gatling's built-in HTTP client supports formUploadBody.
 * This helper builds a chain that:
 *   1. Calls /storage/presigned-form to obtain url + fields (mocked response assumed)
 *   2. Performs a multipart/form-data POST to the returned URL with returned fields and file
 *
 * The helper expects the presigned-form endpoint to return a JSON:
 * { "url": "https://...s3.../bucket", "fields": { "key": "...", "policy": "...", "x-amz-signature": "..."} }
 */
object MultipartUploadHelper {
  def preparePresignedForm(uploadKeySessionKey: String = "uploadKey"): ChainBuilder = {
    exec(session => session.set("uploadKey", s"gatling-${System.currentTimeMillis()}-${scala.util.Random.nextInt(9999)}"))
      .exec(http("Request presigned form")
        .post("/storage/presigned-form")
        .body(StringBody(session => s"""{"key":"${session(uploadKeySessionKey).as[String]}"}""")).asJson
        .check(status.is(200))
        .check(jsonPath("$.url").saveAs("presignedUrl"))
        .check(jsonPath("$.fields").saveAs("presignedFields"))
      )
  }

  def uploadFileFromResource(resourcePath: String = "data/sample-files/sample.txt"): ChainBuilder = {
    exec(session => {
      // extract url/fields from session
      val url = session("presignedUrl").asOption[String].getOrElse("")
      val fieldsJson = session("presignedFields").asOption[io.gatling.commons.validation.Validation[Any]].map(_.toString()).getOrElse("")
      session
    }).exec(http("Storage - multipart direct upload")
      // use formUploadBody to send file and fields. Gatling supports Map[String, Any] bodies.
      .post(session => session("presignedUrl").as[String])
      .formUploadBody(session => {
        // build form fields map; API expects presignedFields to be a map of key/values
        val fields = session("presignedFields").as[Map[String,String]]
        val fileBytes = Files.readAllBytes(Paths.get(resourcePath))
        val byteArrayBody = fileBytes
        // Gatling expects Map[String, Any] where "file" value is called (filename, bytes, contentType)
        // place file under key "file"
        fields + ("file" -> ("sample.txt", byteArrayBody, "text/plain"))
      })
      .check(status.in(200, 201))
    )
  }
}
