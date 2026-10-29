# Ra-Thor Infinite Evolution — VPS Deployment Guide
**Production Deployment for the Infinite Self-Evolution Daemon**

## 1. Server Requirements
- Ubuntu 22.04+ / Debian 12+
- 4GB+ RAM, 2+ CPU cores
- Rust 1.75+ installed

## 2. One-Command Deployment
```bash
git clone https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor.git
cd Ra-Thor
./scripts/launch-infinite-daemon.sh
```

## 3. Systemd Service (Recommended)
Create `/etc/systemd/system/ra-thor-daemon.service`:

```ini
[Unit]
Description=Ra-Thor Infinite Self-Evolution Daemon
After=network.target

[Service]
Type=simple
User=ra-thor
WorkingDirectory=/opt/ra-thor
ExecStart=/usr/local/bin/infinite-evolution-daemon
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## 4. Enable & Start
```bash
sudo systemctl daemon-reload
sudo systemctl enable ra-thor-daemon
sudo systemctl start ra-thor-daemon
sudo journalctl -u ra-thor-daemon -f
```

**Status:** Production-ready. The lattice now runs eternally on your VPS.