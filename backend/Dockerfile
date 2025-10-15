# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY ./requirements.txt /code/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the application code to the working directory
COPY ./app.py /app/app.py
COPY ./utils.py /app/utils.py

# Command to run the application
# Use 0.0.0.0 to make it accessible from outside the container
EXPOSE 8080
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "app:app", "--bind", "0.0.0.0:8080"]
