package common.libraries.kotlin.utils

import com.fasterxml.jackson.annotation.JsonInclude
import com.fasterxml.jackson.core.JsonParseException
import com.fasterxml.jackson.core.type.TypeReference
import com.fasterxml.jackson.databind.DeserializationFeature
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.databind.SerializationFeature
import java.io.InputStream
import kotlin.reflect.KClass

/**
 * JSON utilities using Jackson.
 *
 * Add dependency:
 * implementation("com.fasterxml.jackson.module:jackson-module-kotlin:2.15.+")
 *
 * This object exposes an ObjectMapper configured for production:
 * - registers Kotlin module
 * - ignores unknown properties
 * - writes non-null only
 */
object JsonUtils {
    val mapper: ObjectMapper = ObjectMapper()
        .registerKotlinModule()
        .setSerializationInclusion(JsonInclude.Include.NON_NULL)
        .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false)
        .configure(SerializationFeature.WRITE_DATES_AS_TIMESTAMPS, false)

    inline fun <reified T> fromJson(json: String): T? {
        return try {
            mapper.readValue(json, T::class.java)
        } catch (e: Exception) {
            null
        }
    }

    fun <T : Any> fromJson(json: String, clazz: KClass<T>): T? {
        return try {
            mapper.readValue(json, clazz.java)
        } catch (e: Exception) {
            null
        }
    }

    inline fun <reified T> fromJsonType(json: String, typeRef: TypeReference<T>): T? {
        return try {
            mapper.readValue(json, typeRef)
        } catch (e: Exception) {
            null
        }
    }

    fun toJson(obj: Any): String = mapper.writeValueAsString(obj)

    fun <T> parseStream(stream: InputStream, clazz: Class<T>): T? =
        try {
            mapper.readValue(stream, clazz)
        } catch (e: Exception) {
            null
        }

    /** Safe parse that returns null on parse error but logs optionally. */
    fun safeParse(json: String): Map<String, Any?>? {
        return try {
            mapper.readValue(json, object : TypeReference<Map<String, Any?>>() {})
        } catch (e: JsonParseException) {
            null
        }
    }
}
