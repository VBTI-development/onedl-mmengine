FROM mcr.microsoft.com/devcontainers/python:1-3.10-bookworm

# The base image ships a yarn apt source whose GPG key is no longer valid.
# Removing it here ensures apt-get update succeeds when devcontainer
# features (e.g. nvidia-cuda) are layered on top.
RUN rm -f /etc/apt/sources.list.d/yarn.list \
    && apt-get update \
    && rm -rf /var/lib/apt/lists/*
