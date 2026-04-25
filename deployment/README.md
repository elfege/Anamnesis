# Anamnesis Restart Layer (Phase 7.2b)

Container ↔ host restart trigger using the NVR pattern: tmpfs file as the
sole interface, no Docker socket, no SSH, no sudo from container.

## Architecture

```
┌────────────────────────────────────┐         ┌────────────────────────┐
│  anamnesis-app (Docker container)  │         │  Host (dellserver)      │
│                                    │         │                        │
│  POST /api/anamnesis/config-and-   │         │  systemd unit:         │
│       restart                      │         │  anamnesis-restart-    │
│   ↓                                │         │  watcher.service       │
│  Persists config to MongoDB        │         │   ↓                    │
│   ↓                                │         │  scripts/anamnesis-    │
│  Writes "reboot" to                │         │  restart-watcher.sh    │
│  /dev/shm/anamnesis-restart/       │         │   ↓                    │
│  trigger                           │ tmpfs   │  Polls trigger file    │
│   │                                │ ────────┤   ↓                    │
│   │  bind-mount                    │         │  Sees "reboot"         │
│   ▼                                │         │   ↓                    │
│  /dev/shm/anamnesis-restart/       │         │  Resets file           │
│  trigger                           │         │   ↓                    │
└────────────────────────────────────┘         │  cd $PROJECT_DIR       │
                                                │   ↓                    │
                                                │  ./start.sh            │
                                                │   ↓                    │
                                                │  docker compose up -d  │
                                                │  (picks up new env)    │
                                                └────────────────────────┘
```

## What gets restarted

The watcher runs `./start.sh` from the project root. That's `docker
compose up -d` plus pre-deploy hooks. The new container picks up:

- Updated `.env` values (e.g. `NANOGPT_URLS_RUNPOD` after a RunPod start)
- Persisted config from MongoDB collection `anamnesis_config`
- Any new secrets pulled by `pull_env.sh`

## Install (one-time, on the host)

1. **Bind-mount the tmpfs into the container.** Add to
   `docker-compose.yml` (or `docker-compose.override.yml`):

   ```yaml
   services:
     anamnesis-app:
       volumes:
         - /dev/shm/anamnesis-restart:/dev/shm/anamnesis-restart
   ```

2. **Make the host directory exist with the right permissions.** Add to
   `start.sh` near the top:

   ```bash
   mkdir -p /dev/shm/anamnesis-restart
   chmod 777 /dev/shm/anamnesis-restart
   ```

3. **Install and start the watcher.**

   ```bash
   sudo cp deployment/anamnesis-restart-watcher.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now anamnesis-restart-watcher
   sudo systemctl status anamnesis-restart-watcher
   ```

4. **Restart the Anamnesis stack** so the bind mount takes effect:

   ```bash
   ./start.sh
   ```

5. **Verify.**

   ```bash
   curl http://localhost:3010/api/anamnesis/restart/status
   ```

   Should return `{"available": true, ...}` once the bind mount and
   watcher are both in place.

## Trigger a restart manually (without the UI)

```bash
curl -X POST http://localhost:3010/api/anamnesis/config-and-restart \
  -H 'Content-Type: application/json' \
  -d '{"config": {}, "restart": true}'
```

The container will go down within ~5 seconds.

## Trigger from the UI

The UI's settings tab can call this endpoint when the user changes a
setting that requires restart (e.g. switching the δ² base model, adding
a new secret, swapping `AUTHORIZED_MACHINE_ID`). Currently no UI
component is wired to it — that's a future task.

## Logs

- Watcher: `sudo journalctl -u anamnesis-restart-watcher -f`
- Restart events: `cat ~/0_GENESIS_PROJECT/0_ANAMNESIS/restart_from_app.log`

## Uninstall

```bash
sudo systemctl disable --now anamnesis-restart-watcher
sudo rm /etc/systemd/system/anamnesis-restart-watcher.service
sudo systemctl daemon-reload
```

Then remove the bind mount from `docker-compose.yml`.

## Troubleshooting

- **`/api/anamnesis/restart/status` returns `available=false`**: bind
  mount missing or the watcher hasn't created the trigger file. Check
  the watcher is running (`systemctl status`) and the bind mount is
  declared in compose.

- **`POST /api/anamnesis/config-and-restart` returns 503 with
  PermissionError**: the container can't write to the bind-mounted
  directory. Run `chmod 777 /dev/shm/anamnesis-restart` on the host.

- **Watcher keeps restarting in a loop**: the trigger file may have
  been written with content `"reboot"` and never reset. Check
  `cat /dev/shm/anamnesis-restart/trigger` — should be a status line,
  not the magic string.

- **Container restarts but config not applied**: ensure `start.sh`
  reads from MongoDB collection `anamnesis_config` and exports the
  values as env vars before `docker compose up -d`.
