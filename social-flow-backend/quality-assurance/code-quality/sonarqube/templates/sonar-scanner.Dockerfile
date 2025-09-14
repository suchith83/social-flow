# Build a small reproducible container with sonar-scanner CLI (scanner-cli)
FROM openjdk:17-jdk-slim

ARG SCANNER_VERSION=5.12.0.7126
ENV SONAR_SCANNER_HOME=/opt/sonar-scanner
ENV PATH="${SONAR_SCANNER_HOME}/bin:${PATH}"

# Install curl/unzip
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl unzip && \
    rm -rf /var/lib/apt/lists/*

# Download SonarScanner CLI
RUN mkdir -p ${SONAR_SCANNER_HOME} && \
    curl -L "https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/sonar-scanner-cli-${SCANNER_VERSION}.zip" -o /tmp/scanner.zip && \
    unzip /tmp/scanner.zip -d /opt && \
    mv /opt/sonar-scanner-${SCANNER_VERSION} ${SONAR_SCANNER_HOME} && \
    rm /tmp/scanner.zip

# Add a default wrapper script that forwards environment variables to sonar-scanner
COPY sonar-scanner-wrapper.sh /usr/local/bin/sonar-scanner-wrapper
RUN chmod +x /usr/local/bin/sonar-scanner-wrapper

ENTRYPOINT ["sonar-scanner-wrapper"]


# #!/usr/bin/env bash
# # sonar-scanner-wrapper.sh
# # A wrapper that fills common parameters from env variables and executes sonar-scanner

# set -euo pipefail

# # Map env variables to scanner properties
# # Required:
# #  - SONAR_HOST_URL (e.g. https://sonarcloud.io or http://localhost:9000)
# #  - SONAR_TOKEN (user token with execute analysis rights)

# if [ -z "${SONAR_HOST_URL:-}" ]; then
#   echo "SONAR_HOST_URL is not set. Exiting."
#   exit 1
# fi
# if [ -z "${SONAR_TOKEN:-}" ]; then
#   echo "SONAR_TOKEN is not set. Exiting."
#   exit 1
# fi

# ARGS=()
# ARGS+=("-Dsonar.host.url=${SONAR_HOST_URL}")
# ARGS+=("-Dsonar.login=${SONAR_TOKEN}")

# # Optional fields
# if [ -n "${SONAR_PROJECT_KEY:-}" ]; then
#   ARGS+=("-Dsonar.projectKey=${SONAR_PROJECT_KEY}")
# fi
# if [ -n "${SONAR_PROJECT_NAME:-}" ]; then
#   ARGS+=("-Dsonar.projectName=${SONAR_PROJECT_NAME}")
# fi
# if [ -n "${GITHUB_REF_NAME:-}" ]; then
#   ARGS+=("-Dsonar.branch.name=${GITHUB_REF_NAME}")
# fi
# if [ -n "${GITHUB_HEAD_REF:-}" ]; then
#   ARGS+=("-Dsonar.pullrequest.key=${GITHUB_PR_NUMBER:-}")
#   ARGS+=("-Dsonar.pullrequest.branch=${GITHUB_HEAD_REF}")
#   ARGS+=("-Dsonar.pullrequest.base=${GITHUB_BASE_REF:-}")
# fi

# # Append any CLI args passed to container
# ARGS+=("$@")

# echo "Running sonar-scanner with args: ${ARGS[*]}"
# exec sonar-scanner "${ARGS[@]}"
