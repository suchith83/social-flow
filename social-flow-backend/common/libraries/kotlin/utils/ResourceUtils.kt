package common.libraries.kotlin.utils

import java.io.File
import java.io.InputStream
import java.nio.file.Files
import java.nio.file.StandardCopyOption

/**
 * File / resource related helpers for reading resources, atomic writes, temp files.
 */
object ResourceUtils {
    /** Read resource from classpath into string or null if missing. */
    fun readResourceAsString(path: String, classLoader: ClassLoader = Thread.currentThread().contextClassLoader): String? {
        val stream = classLoader.getResourceAsStream(path) ?: return null
        return stream.bufferedReader(Charsets.UTF_8).use { it.readText() }
    }

    /** Copy InputStream to a destination File atomically (write to temp and move). */
    fun atomicWrite(input: InputStream, destination: File) {
        val tmp = File(destination.parentFile, "${destination.name}.${StringUtils.randomId(6)}.tmp")
        tmp.outputStream().use { out -> input.copyTo(out) }
        Files.move(tmp.toPath(), destination.toPath(), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
    }

    /** Ensure directory exists; create parents if needed. */
    fun ensureDir(dir: File) {
        if (!dir.exists()) {
            val created = dir.mkdirs()
            if (!created && !dir.exists()) throw IllegalStateException("Unable to create directory ${dir.absolutePath}")
        }
    }
}
