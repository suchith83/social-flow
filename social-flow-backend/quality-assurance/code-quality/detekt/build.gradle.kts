// Gradle build script for Detekt configuration
plugins {
    // Kotlin JVM plugin
    kotlin("jvm") version "1.9.10"

    // Detekt plugin
    id("io.gitlab.arturbosch.detekt") version "1.23.1"
}

repositories {
    mavenCentral()
}

dependencies {
    // Detekt formatting rules
    detektPlugins("io.gitlab.arturbosch.detekt:detekt-formatting:1.23.1")

    // Kotlin standard library
    implementation(kotlin("stdlib"))

    // Needed for custom rules
    implementation("io.gitlab.arturbosch.detekt:detekt-api:1.23.1")
    testImplementation("io.gitlab.arturbosch.detekt:detekt-test:1.23.1")

    // Unit testing
    testImplementation("org.jetbrains.kotlin:kotlin-test")
    testImplementation("junit:junit:4.13.2")
}

tasks {
    detekt {
        // Config files
        config.setFrom(files("$projectDir/detekt-config.yml"))
        buildUponDefaultConfig = true

        // Baseline file to suppress known issues
        baseline.set(file("$projectDir/detekt-baseline.xml"))

        // Where to scan
        source.setFrom(
            files(
                "src/main/kotlin",
                "src/test/kotlin"
            )
        )

        parallel = true // Speed up analysis
    }

    // Run detekt automatically with build
    check {
        dependsOn(detekt)
    }
}
