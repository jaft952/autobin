# Auto-start server.py on Pi boot

One-time setup (run on the Pi):

```
sudo cp ~/Desktop/autobin/deploy/autobin-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable autobin-server
sudo systemctl start autobin-server
```

After this, `server.py` starts automatically on every Pi boot — no SSH needed.

Useful commands:

```
sudo systemctl status autobin-server    # check it's running
sudo systemctl restart autobin-server   # after pulling code changes
journalctl -u autobin-server -f         # live logs
sudo systemctl stop autobin-server      # stop it
```

If the repo path or venv path ever changes, edit
`/etc/systemd/system/autobin-server.service` and run
`sudo systemctl daemon-reload && sudo systemctl restart autobin-server`.
