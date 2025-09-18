name := "gatling-perf-suite"

version := "0.1.0"

scalaVersion := "2.13.12" // Gatling 3.10+ supports Scala 2.13

libraryDependencies ++= Seq(
  "io.gatling" % "gatling-core" % "3.10.4" % "test",
  "io.gatling" % "gatling-http" % "3.10.4" % "test",
  "com.typesafe" % "config" % "1.4.2",
  "org.scalatest" %% "scalatest" % "3.2.17" % "test",
  "com.github.tomakehurst" % "wiremock-jre8-standalone" % "2.35.0" % "test"
)

enablePlugins(GatlingPlugin)

resolvers += Resolver.sonatypeRepo("releases")
