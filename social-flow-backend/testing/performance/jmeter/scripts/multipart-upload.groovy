/**
 * multipart-upload.groovy
 *
 * Example JSR223 (Groovy) sampler that:
 * 1. Requests a presigned form/URL from the app at /storage/presigned-form
 * 2. Performs a multipart/form-data POST to the returned URL (S3-style presigned POST)
 *
 * Expects variables:
 * - upload.key (sample key to request)
 * - base_url
 * - storage_bucket (optional)
 * - file_path (path on JMeter host to local file to upload)
 *
 * Note: Running file uploads from JMeter requires the file to be available on the machine executing JMeter.
 * In Docker, mount the local files directory into the container.
 */

import org.apache.http.client.methods.HttpPost
import org.apache.http.impl.client.HttpClients
import org.apache.http.entity.StringEntity
import groovy.json.JsonSlurper
import org.apache.http.entity.mime.MultipartEntityBuilder
import org.apache.http.entity.ContentType
import org.apache.http.util.EntityUtils
import org.apache.jorphan.logging.LoggingManager

def log = LoggingManager.getLoggerForClass()
def baseUrl = props.getProperty('base_url') ?: vars.get('base_url') ?: 'http://localhost:3000/api'
def key = vars.get('upload.key') ?: 'perf-upload-' + System.currentTimeMillis()
def filePath = vars.get('file_path') ?: props.getProperty('storage_sample_file') ?: './data/sample.txt'
def bucket = props.getProperty('storage_bucket') ?: vars.get('storage_bucket')

log.info("multipart-upload: requesting presigned form for key=${key}")

def httpclient = HttpClients.createDefault()
def postReq = new HttpPost("${baseUrl}/storage/presigned-form")
postReq.setHeader("Content-Type", "application/json")
def bodyJson = ['key': key]
if (bucket) bodyJson['bucket'] = bucket
postReq.setEntity(new StringEntity(new groovy.json.JsonBuilder(bodyJson).toPrettyString(), "UTF-8"))

def resp = httpclient.execute(postReq)
def code = resp.getStatusLine().getStatusCode()
def respBody = EntityUtils.toString(resp.getEntity(), "UTF-8")
if (code < 200 || code >= 300) {
    log.error("Presigned form request failed: ${code} - ${respBody}")
    SampleResult.setResponseCode(code.toString())
    SampleResult.setSuccessful(false)
    SampleResult.setResponseMessage("Presigned form request failed")
    return
}

def json = new JsonSlurper().parseText(respBody)
def uploadUrl = json.url
def formFields = json.fields ?: [:]

log.info("Got presigned url: ${uploadUrl} fields: ${formFields.keySet()}")

// Build multipart
def builder = MultipartEntityBuilder.create()
formFields.each { k, v ->
    builder.addTextBody(k.toString(), v.toString())
}

// attach the file as the 'file' field (S3 presigned POST expects 'file')
def file = new java.io.File(filePath)
if (!file.exists()) {
    log.error("File not found: ${filePath}")
    SampleResult.setSuccessful(false)
    SampleResult.setResponseMessage("File not found: ${filePath}")
    return
}
builder.addBinaryBody('file', file, ContentType.DEFAULT_BINARY, file.getName())

def uploadPost = new HttpPost(uploadUrl)
uploadPost.setEntity(builder.build())

def uploadResp = httpclient.execute(uploadPost)
def uploadCode = uploadResp.getStatusLine().getStatusCode()
def uploadBody = EntityUtils.toString(uploadResp.getEntity(), "UTF-8")
if (uploadCode >= 200 && uploadCode < 300) {
    log.info("Upload success code=${uploadCode}")
    SampleResult.setResponseCode(uploadCode.toString())
    SampleResult.setSuccessful(true)
    SampleResult.setResponseData(uploadBody, 'UTF-8')
} else {
    log.error("Upload failed: ${uploadCode} - ${uploadBody}")
    SampleResult.setResponseCode(uploadCode.toString())
    SampleResult.setSuccessful(false)
    SampleResult.setResponseMessage("Upload failed")
}
