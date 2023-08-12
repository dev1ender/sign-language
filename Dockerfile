# Use the official Nginx image as the base image
FROM nginx:latest

# Install uWSGI and Python
RUN apt-get update && apt-get install -y uwsgi uwsgi-plugin-python3 python3-pip python3-venv

# Create a directory for the Flask app
WORKDIR /app

# Copy the Flask app files into the container
COPY . .

COPY nginx/nginx.conf /etc/nginx/conf.d/

# Create a virtual environment
RUN python3 -m venv venv

# Activate the virtual environment and install Flask and other dependencies
RUN . venv/bin/activate && pip install -r requirements.txt

# Expose the ports
EXPOSE 80

# Start the uWSGI server
CMD ["uwsgi", "--ini", "uwsgi.ini"]
