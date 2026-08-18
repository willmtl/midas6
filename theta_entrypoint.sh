#!/bin/sh
# Launch the Theta Terminal with creds from env, then expose its REST API on 0.0.0.0:8080 for the
# backend container (the terminal may bind 127.0.0.1:25510 only, so socat forwards it).
set -e
: "${THETADATA_API_KEY:?set THETADATA_API_KEY}"
: "${THETADATA_PASSWORD:?set THETADATA_PASSWORD in .env (your thetadata.net account password)}"

printf '%s\n%s\n' "$THETADATA_API_KEY" "$THETADATA_PASSWORD" > /opt/theta/creds.txt
echo "[theta] launching terminal (user=${THETADATA_API_KEY%%_*}_...)"
java -jar /opt/theta/ThetaTerminal.jar --creds-file=/opt/theta/creds.txt &
TPID=$!

# wait for the REST port to come up on loopback, then forward it to 0.0.0.0:8080
for i in $(seq 1 30); do
  if (echo > /dev/tcp/127.0.0.1/25510) 2>/dev/null; then break; fi
  sleep 1
done
echo "[theta] forwarding 0.0.0.0:8080 -> 127.0.0.1:25510 (REST)"
socat TCP-LISTEN:8080,fork,reuseaddr TCP:127.0.0.1:25510 &

wait "$TPID"
