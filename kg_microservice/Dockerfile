# Choose a base image
FROM ubuntu:latest

# Install dependencies (Java, Python, etc.)
RUN apt-get update \
    && apt-get install -y default-jdk

# Set up Stanford CoreNLP
COPY stanford-corenlp /stanford-corenlp
WORKDIR /stanford-corenlp

# Expose the appropriate port
EXPOSE 9000

# Define the startup command
CMD ["java", "-Xmx4g", "-cp", ".", "edu.stanford.nlp.pipeline.StanfordCoreNLPServer", "-threads", "1", "-timeout", "60000"]
