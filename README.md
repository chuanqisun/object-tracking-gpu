# Object Tracking GPU Accelerated

<img alt="puck-tracking" src="https://github.com/user-attachments/assets/17aab931-562b-4551-bc71-598f9f0d3e22" />

- Object tracking: YOLO26
- Hand tracking: mediapipe

See [Non-device specific version](https://github.com/chuanqisun/object-tracking)

```bash
# Activate python environment
source .venv/bin/activate

# Start backend
uv run python main.py

# Start frontend
uvx livereload web
```

## How to set custom Bluetooth labels in Linux

1. Open a terminal and enter the Bluetooth control shell:
   ```bash
   bluetoothctl
   ```
2. List paired devices to find the MAC address of the device:

   ```text
   devices
   ```

   _(Note the MAC address in the format `XX:XX:XX:XX:XX:XX`)_

3. Set the alias:
   ```text
   alias XX:XX:XX:XX:XX:XX "My Custom Name"
   ```
4. Exit the shell:
   ```text
   exit
   ```
