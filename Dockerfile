# This is for CaptainRover

# STEP 1: Install base image. Optimized for Python.
FROM python:3.8-slim-buster

# STEP 2: Copy the source code in the current directory to the container.  Store it in a folder named /app.
ADD . /app

# STEP 3: Set working directory to /app so we can execute commands in it
WORKDIR /app

# STEP 4: Install necessary requirements (Voila, Jax etc)
RUN pip install -r requirements.txt 

# STEP 5: Declare environment variables
ENV PORT "8080"

# STEP 6: Expose the port that Voila is running on
EXPOSE ${PORT}

# STEP 7: Run Voila!
CMD ["voila", "techni.ipynb", "--no-browser", "--port=8080"]
