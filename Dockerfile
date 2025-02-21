# Use the official Python base image
FROM python:3.10.14-bookworm

# Set a working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code to the container
COPY . .

# Expose the port the app will run on (adjust if necessary)
EXPOSE 9001

# Command to run your app
CMD ["python", "app.py"]