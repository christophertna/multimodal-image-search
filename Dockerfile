# Dockerfile:
# Defines the container IMAGE for the multimodal image search app
#
#
# What is a Dockerfile?
#   Basically a recipe that tells Docker how to build a container IMAGE
#
#   -   An IMAGE is a snapshot of an environment (OS + dependencies + the code)
#
#   -   A CONTAINER is a running instance of that image (lightweight VM
#       You build the image once then run it anywhere)
#
#
# Build & run:
#   docker compose build
#   docker compose up


# FROM (base image)
#
# Every Dockerfile starts with a base IMAGE to build on top of:
# For this project using Python 3.12 slim image:
#   - "slim": it strips out unnecessary OS packages, keeping the IMAGE small
#   - Use 3.12 specifically because PyTorch CUDA builds don't support 3.13 yet
#
# (Kinda like choosing which OS + Python version the app runs on)
FROM python:3.12-slim


# ENV (environment variables)
#
# Environment variables set inside the container
#
# PYTHONDONTWRITEBYTECODE=1:
#   Prevents Python from writing .pyc bytecode cache files to disk
#   (Not needed inside a container since never reuse cached bytecode)
#
# PYTHONUNBUFFERED=1:
#   Forces Python's stdout/stderr to be unbuffered, meaning print() statements
#   and logs appear in the terminal immediately instead of being held in a buffer
#   (Without this Docker logs can appear delayed/out of order)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


# WORKDIR (set the working directory inside the container)
# 
# All subsequent commands (COPY, RUN, CMD) run relative to this path:
# If the directory doesnt exist then Docker creates it automatically
# (basically "cd /app" for all subsequent commands)
WORKDIR /app


# COPY requirements first (layer caching optimization)
# 
# Docker builds images in layers, 1 layer per instruction
# Layers are cached so if nothing changed, then Docker reuses the cached layer
#
# By copying requirements.txt BEFORE the rest of the code, we ensure that
# the pip install layer is only re-run when requirements.txt changes
#
# If you only edit app.py, Docker reuses the cached pip install layer
# and skips straight to copying your code (much faster rebuilds)
#
COPY requirements.txt .


# RUN (execute commands during image build)
# 
# RUN commands execute at BUILD time (docker compose build), not at runtime:
# Each RUN instruction creates a new layer in the image
#
# We chain commands with "&&"" to keep them in a single layer (smaller image)
#
# "--no-cache-dir" tells pip NOT to store the download cache inside the image,
# since we will never reuse it (keeps the image size smaller)
#
# *** Note: Install the CPU version of torch here for portability and simplicity ***
#
# Anyone can run the container regardless of GPU or not
# (If you want GPU support inside Docker, need nvidia-docker, which requires more advanced setup)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt


# COPY (copy project files into container)
# 
# Copies everything from local project folder into "/app" inside the container
# (Files listed in .dockerignore are excluded)
#
# runs AFTER pip install so code changes dont invalidate the pip cache layer
COPY . .


# EXPOSE (document port the app listens on)
#
# Streamlit runs on port 8501 by default
# EXPOSE doesnt actually open the port, just documentation for docker compose,
# which does the actual port binding (see docker-compose.yml)
EXPOSE 8501


# CMD (command that runs when the container starts)
# 
# CMD runs at CONTAINER START time (docker compose up), NOT at build time
# (Only 1 CMD is allowed per Dockerfile, its the app's entry point)
#
# Flags explained:
#   "--server.address=0.0.0.0":  makes Streamlit accessible from OUTSIDE the container
#                                (default only listens on localhost, which is
#                                unreachable from host machine)
#
#   "--server.port=8501":         explicit port (matches EXPOSE above)
#
#   "--server.fileWatcherType=none": disables Streamlit's file watcher inside Docker
#                                    (it uses 'inotify' which can cause issues in containers)
CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.fileWatcherType=none"]