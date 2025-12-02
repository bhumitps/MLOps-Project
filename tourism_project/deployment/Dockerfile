# Use a minimal base image with Python 3.9 installed
FROM python:3.9

# Set environment variables for the application
ENV APP_HOME=/app

# Set up the non-root user (following best practice from source notebooks [cite: 1, 2])
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# Set the working directory inside the container
WORKDIR $HOME/$APP_HOME

# Copy application files and install dependencies
# We assume the build context (which contains requirements.txt and app.py) is available
COPY requirements.txt ./
RUN pip3 install -r requirements.txt

COPY --chown=user . $HOME/$APP_HOME

# Define the command to run the Streamlit app on port "8501" and make it accessible externally
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
