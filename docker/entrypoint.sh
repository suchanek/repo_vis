#!/bin/sh
# Start a virtual display, then run pkg-visualizer against /pkg.
#
# Any arguments are passed to pkg-visualizer. With --headless the scene is
# written to /out and the container exits; otherwise the GUI is served over
# noVNC on port 6080.
set -e

# The container cannot see the host directory name behind the /pkg mount,
# so the output name comes from PKG_NAME (docker run -e PKG_NAME=...).
PKG_NAME=${PKG_NAME:-package}

Xvfb "$DISPLAY" -screen 0 "$SCREEN_GEOMETRY" -nolisten tcp >/dev/null 2>&1 &
# Wait for the display socket before starting X clients.
i=0
while [ ! -e "/tmp/.X11-unix/X${DISPLAY#:}" ] && [ $i -lt 50 ]; do
    sleep 0.1
    i=$((i + 1))
done

case " $* " in
    *" --headless "*)
        exec pkg-visualizer -k /pkg -s "/out/$PKG_NAME" "$@"
        ;;
esac

# Start every window maximized so the app fits the virtual screen.
mkdir -p "$HOME/.fluxbox"
printf '[app] (name=.*)\n  [Maximized] {yes}\n[end]\n' > "$HOME/.fluxbox/apps"
fluxbox >/dev/null 2>&1 &
x11vnc -display "$DISPLAY" -forever -shared -nopw -quiet -rfbport 5900 \
    -localhost >/dev/null 2>&1 &
websockify --web /usr/share/novnc 6080 localhost:5900 >/dev/null 2>&1 &

echo "pkg-visualizer GUI: http://localhost:6080/vnc.html?autoconnect=1&resize=scale"

W=${SCREEN_GEOMETRY%%x*}
H=${SCREEN_GEOMETRY#*x}
H=${H%%x*}
exec pkg-visualizer -k /pkg -s "$PKG_NAME" -w "$W" -t "$H" "$@"
