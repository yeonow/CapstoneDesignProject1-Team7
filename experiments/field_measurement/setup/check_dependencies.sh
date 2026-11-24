
#!/usr/bin/env bash

# Check dependencies required for RSSI field measurements.

commands=(
    python3
    tshark
    capinfos
    iw
    ip
    nmcli
    lsusb
    vcgencmd
)

missing=0

echo "=== RSSI Measurement Dependency Check ==="

for cmd in "${commands[@]}"; do
    if command -v "$cmd" >/dev/null 2>&1; then
        printf "[OK] %s\n" "$cmd"
    else
        printf "[MISSING] %s\n" "$cmd"
        missing=$((missing + 1))
    fi
done

echo "========================================="

if [ "$missing" -eq 0 ]; then
    echo "All required dependencies are available."
    exit 0
else
    echo "$missing required dependencies are missing."
    exit 1
fi
