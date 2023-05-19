# Choose a base image
FROM ubuntu:latest

# Install dependencies (Java, Python, etc.)
RUN apt-get update \
    && apt-get install -y default-jdk

# Set up Stanford CoreNLP
COPY stanford-corenlp /stanford-corenlp
COPY bootstrap_server.sh /var/opt/bootstrap_server.sh
# WORKDIR /var/opt
WORKDIR /stanford-corenlp

# Expose the appropriate port
EXPOSE 9000

# Define the startup command
# CMD ["./bootstrap_server.sh"]
CMD ["java", "-Xmx4g", "-cp", "/stanford-corenlp/*", "edu.stanford.nlp.pipeline.StanfordCoreNLPServer", "-threads", "1", "-timeout", "60000"]
