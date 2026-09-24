# pkg_visualizer in a container.
#
# Default: the Qt GUI runs on a virtual display and is served to the browser
# through noVNC at http://localhost:6080/vnc.html
# With --headless: render off-screen, write the file to /out, and exit.
#
# Mount the package to visualize at /pkg and an output directory at /out.
# See docker/entrypoint.sh and the Docker section of README.md.
#
# Multi-arch (linux/amd64, linux/arm64). PyQt5 publishes Linux wheels for
# x86_64 only, so it comes from Debian's python3-pyqt5 instead, which exists
# for both architectures; the venv sees it through --system-site-packages.
# arm64 runs natively under Apple's `container` and Docker Desktop on Apple
# Silicon. The rest installs from the pyproject.toml ranges, not poetry.lock:
# the lock pins vtk 9.4.2, and VTK ships arm64 Linux wheels only from 9.5.

FROM debian:trixie-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH=/opt/venv/bin:$PATH \
    DISPLAY=:99 \
    SCREEN_GEOMETRY=1920x1200x24

# Python and PyQt5, virtual display, VNC, noVNC, a small window manager,
# and the GL/X libraries the VTK wheel loads.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pyqt5 \
        tini xvfb x11vnc novnc websockify fluxbox \
        libgl1 libgl1-mesa-dri libegl1 libxrender1 libxext6 libsm6 libxi6 \
        libxcursor1 libxinerama1 libxrandr2 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv --system-site-packages /opt/venv

WORKDIR /app
# Dependencies first, so source edits don't reinstall them.
COPY pyproject.toml ./
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']))" > /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt

COPY README.md LICENSE ./
COPY pkg_visualizer ./pkg_visualizer
RUN pip install --no-deps . && pip check

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh && mkdir -p /pkg /out

WORKDIR /out
EXPOSE 6080
ENTRYPOINT ["tini", "--", "entrypoint.sh"]
