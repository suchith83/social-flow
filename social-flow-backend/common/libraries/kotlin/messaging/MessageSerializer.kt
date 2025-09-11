package common.libraries.kotlin.messaging

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.KotlinModule
import com.fasterxml.jackson.module.kotlin.readValue
import java.nio.charset.StandardCharsets

/**
 * Simple JSON serializer using Jackson. You can replace with protobuf/avro by implementing same interface.
 */
interface MessageSerializer {
    fun <T> serialize(payload: T): ByteArray
    fun <T> deserialize(bytes: ByteArray, clazz: Class<T>): T
}

/**
 * Jackson-based implementation.
 */
class JacksonMessageSerializer(private val mapper: ObjectMapper = defaultMapper()) : MessageSerializer {
    override fun <T> serialize(payload: T): ByteArray =
        mapper.writeValueAsString(payload).toByteArray(StandardCharsets.UTF_8)

    override fun <T> deserialize(bytes: ByteArray, clazz: Class<T>): T =
        mapper.readValue(bytes, clazz)

    companion object {
        fun defaultMapper(): ObjectMapper {
            val m = ObjectMapper()
            m.registerModule(KotlinModule.Builder().build())
            // configure as needed (fail on unknown, snake_case, etc.)
            return m
        }
    }
}
