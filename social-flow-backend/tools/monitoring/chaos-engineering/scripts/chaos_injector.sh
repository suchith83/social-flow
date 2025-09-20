#!/bin/bash
# chaos_injector.sh - System-level chaos experiments runner

set -e

EXPERIMENT=$1
DURATION=${2:-60}

case $EXPERIMENT in
  cpu)
    echo "🔥 Injecting CPU stress for $DURATION seconds..."
    stress --cpu 4 --timeout "$DURATION"
    ;;
  memory)
    echo "🔥 Injecting memory stress for $DURATION seconds..."
    stress --vm 2 --vm-bytes 512M --timeout "$DURATION"
    ;;
  disk)
    echo "🔥 Filling disk for $DURATION seconds..."
    dd if=/dev/zero of=/tmp/chaos_fill bs=1M count=1024 &
    PID=$!
    sleep "$DURATION"
    kill $PID || true
    rm -f /tmp/chaos_fill
    ;;
  network)
    echo "🌐 Adding network latency..."
    tc qdisc add dev eth0 root netem delay 200ms
    sleep "$DURATION"
    tc qdisc del dev eth0 root netem
    ;;
  *)
    echo "❌ Unknown experiment: $EXPERIMENT"
    exit 1
    ;;
esac

echo "✅ Chaos experiment $EXPERIMENT finished."
