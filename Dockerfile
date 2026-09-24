# pkg_visualizer in a container.
#
# Default: the Qt GUI runs on a virtual display and is served to the browser
# through noVNC at http://localhost:6080/vnc.html
# With --headless: render off-screen, write the file to /out, and exit.
#
# Mount the package to visualize at /pkg and an output directory at /out.
# See docker/entrypoint.sh and the Docker section of README.md.

# PyQt5 publishes Linux wheels for x86_64 only, so the image is amd64;
# Docker Desktop runs it under Rosetta on Apple Silicon.
FROM --platform=linux/amd64 python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    DISPLAY=:99 \
    SCREEN_GEOMETRY=1920x1200x24

# Virtual display, VNC, noVNC, a small window manager, Mesa for VTK,
# and the X libraries the Qt xcb platform plugin loads.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tini xvfb x11vnc novnc websockify fluxbox \
        libgl1 libgl1-mesa-dri libegl1 libglib2.0-0 libdbus-1-3 \
        libfontconfig1 libxrender1 libxext6 libsm6 libxi6 libxkbcommon-x11-0 \
        libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
        libxcb-render-util0 libxcb-xinerama0 libxcb-xfixes0 libxcb-shape0 \
        libxcb-cursor0 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "poetry>=2,<3"

WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root --no-interaction

COPY README.md LICENSE ./
COPY pkg_visualizer ./pkg_visualizer
RUN poetry install --only main --no-interaction

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh && mkdir -p /pkg /out

WORKDIR /out
EXPOSE 6080
ENTRYPOINT ["tini", "--", "entrypoint.sh"]
