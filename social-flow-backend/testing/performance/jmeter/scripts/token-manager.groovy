/**
 * token-manager.groovy
 *
 * JSR223 Groovy helper: token caching across threads in a thread-safe manner.
 * Use as a PreProcessor:
 * - it checks ctx.getThreadGroup() / or static cache for token
 * - if token missing or near expiry, perform HTTP login (via HttpClient) and store token in JMeter properties
 *
 * NOTE: JMeter properties are global across the test run (synchronized across threads),
 * so we store cached token in props (org.apache.jmeter.util.JMeterUtils).
 *
 * This script expects the test plan to expose variables:
 * - base_url (JMeter property or variable)
 * - test_user, test_password
 *
 * Example usage in JMeter:
 * - Add a JSR223 PreProcessor to a HTTP Sampler and paste this script or use "filename" to load it.
 */

import org.apache.jmeter.util.JMeterUtils
import org.apache.jorphan.logging.LoggingManager
import org.apache.log.Logger
import groovy.json.JsonSlurper
import org.apache.http.client.methods.HttpPost
import org.apache.http.entity.StringEntity
import org.apache.http.impl.client.HttpClients
import org.apache.http.util.EntityUtils

Logger log = LoggingManager.getLoggerForClass()
def props = ctx.getEngine().getProperties() // global props
// fallback to JMeter properties
if (props == null) props = JMeterUtils.getJMeterProperties()

// helper to get property or variable
def getPropOrVar = { name ->
    def v = vars.get(name)
    if (v == null || v.trim() == "") v = props.getProperty(name)
    return v
}

def baseUrl = getPropOrVar('base_url') ?: 'http://localhost:3000/api'
def user = getPropOrVar('test_user') ?: 'admin'
def pwd = getPropOrVar('test_password') ?: 'password123'

// token cache key
def tokenKey = 'PERF_AUTH_TOKEN'
def tokenExpiryKey = 'PERF_AUTH_TOKEN_EXPIRY' // ms epoch

synchronized(this.class) {
    def token = props.getProperty(tokenKey)
    def expiry = props.getProperty(tokenExpiryKey)?.toLong() ?: 0L
    def now = System.currentTimeMillis()

    if (token == null || now >= expiry - 30000) { // refresh if within 30s of expiry
        log.info("Token missing or expiring soon, requesting new token for ${user}")
        try {
            def httpclient = HttpClients.createDefault()
            def post = new HttpPost("${baseUrl}/auth/login")
            post.setHeader("Content-Type", "application/json")
            post.setEntity(new StringEntity("{\"username\":\"${user}\",\"password\":\"${pwd}\"}", "UTF-8"))

            def resp = httpclient.execute(post)
            def code = resp.getStatusLine().getStatusCode()
            def body = EntityUtils.toString(resp.getEntity(), "UTF-8")
            if (code >= 200 && code < 300) {
                def json = new JsonSlurper().parseText(body)
                def t = json.token ?: json.access_token ?: null
                def expires_in = (json.expires_in ?: json.expiresIn ?: 3600) as long
                if (t) {
                    props.setProperty(tokenKey, t.toString())
                    props.setProperty(tokenExpiryKey, (now + (expires_in * 1000)).toString())
                    token = t.toString()
                    log.info("Acquired new token (len=${token.length()}) expires_in=${expires_in}s")
                } else {
                    log.warn("Token not present in login response: ${body}")
                }
            } else {
                log.warn("Login failed code=${code} body=${body}")
            }
        } catch (e) {
            log.error("Error acquiring token: ${e}", e)
        }
    } else {
        // token is valid
        log.debug("Reusing cached token (expires at ${expiry})")
    }

    // finally, set Authorization header variable for the sampler to consume
    def finalToken = props.getProperty(tokenKey)
    if (finalToken) {
        vars.put('auth_header_value', "Bearer ${finalToken}")
    } else {
        vars.put('auth_header_value', "")
    }
}
