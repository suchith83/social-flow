# Base image: official JMeter + plugins
FROM openjdk:17-jdk-slim

ENV JMETER_VERSION 5.6.3
ENV JMETER_HOME /opt/jmeter
ENV PATH $JMETER_HOME/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget unzip ca-certificates curl procps && \
    rm -rf /var/lib/apt/lists/*

# Install JMeter
WORKDIR /opt
RUN wget -q https://archive.apache.org/dist/jmeter/binaries/apache-jmeter-${JMETER_VERSION}.tgz && \
    tar -xzf apache-jmeter-${JMETER_VERSION}.tgz && \
    mv apache-jmeter-${JMETER_VERSION} jmeter && \
    rm apache-jmeter-${JMETER_VERSION}.tgz

ENV JMETER_HOME /opt/jmeter
RUN mkdir -p $JMETER_HOME/bin/ext

# Install JMeter Plugins Manager (optional but recommended)
RUN wget -q -O $JMETER_HOME/bin/ext/jmeter-plugins-manager.jar https://jmeter-plugins.org/get/ && \
    java -cp $JMETER_HOME/bin/ext/jmeter-plugins-manager.jar org.jmeterplugins.repository.PluginManagerCMDInstaller

# Copy test assets
WORKDIR /work
COPY . /work

# Make entrypoint script executable
COPY bin/jmeter-entrypoint.sh /usr/local/bin/jmeter-entrypoint.sh
RUN chmod +x /usr/local/bin/jmeter-entrypoint.sh

# Install node for helper scripts if needed (optional)
# RUN apt-get update && apt-get install -y nodejs npm

ENTRYPOINT ["/usr/local/bin/jmeter-entrypoint.sh"]
CMD ["jmeter", "-v"]
