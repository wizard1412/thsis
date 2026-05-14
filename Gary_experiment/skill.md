# Remote Server SSH Connection via PuTTY Key

## Server Info

- Host: `140.116.155.8`
- Port: `8801`
- User: `gw`
- Key: `C:\Users\user\Documents\puttykey\mac.ppk`

## Tool Used

`plink.exe` from PuTTY — supports `.ppk` keys natively, no conversion needed.

Location: `C:\Program Files\PuTTY\plink.exe`

## Important: Use `bash -l -c` for Full PATH

The server's tools (tmux, python, conda…) are installed via Homebrew at `/opt/homebrew/bin`.
A non-login shell won't find them. Always wrap commands with `bash -l -c '...'`.

```cmd
"C:\Program Files\PuTTY\plink.exe" -i "C:\Users\user\Documents\puttykey\mac.ppk" -P 8801 -batch gw@140.116.155.8 "bash -l -c '<command>'"
```

### Examples

```cmd
REM Check connection
"C:\Program Files\PuTTY\plink.exe" -i "C:\Users\user\Documents\puttykey\mac.ppk" -P 8801 -batch gw@140.116.155.8 "bash -l -c 'echo connected && hostname && whoami'"

REM Check Python version
"C:\Program Files\PuTTY\plink.exe" -i "C:\Users\user\Documents\puttykey\mac.ppk" -P 8801 -batch gw@140.116.155.8 "bash -l -c 'which python && python --version'"

REM Run a Python script already on the server
"C:\Program Files\PuTTY\plink.exe" -i "C:\Users\user\Documents\puttykey\mac.ppk" -P 8801 -batch gw@140.116.155.8 "bash -l -c 'python /path/to/script.py'"

REM Run a long job in the background (nohup) — safe to close window or sleep
"C:\Program Files\PuTTY\plink.exe" -i "C:\Users\user\Documents\puttykey\mac.ppk" -P 8801 -batch gw@140.116.155.8 "bash -l -c 'nohup python /path/to/script.py > output.log 2>&1 &'"
```

## Run Persistent Jobs with tmux

tmux keeps the job alive even after closing the window or sleeping the laptop.
tmux is at `/opt/homebrew/bin/tmux` (version 3.6a), loaded via `bash -l`.

```cmd
REM Start a new tmux session and run script inside it
"C:\Program Files\PuTTY\plink.exe" -i "C:\Users\user\Documents\puttykey\mac.ppk" -P 8801 -batch gw@140.116.155.8 "bash -l -c 'tmux new-session -d -s myjob \"python /path/to/script.py\"'"

REM Check if session is still running
"C:\Program Files\PuTTY\plink.exe" -i "C:\Users\user\Documents\puttykey\mac.ppk" -P 8801 -batch gw@140.116.155.8 "bash -l -c 'tmux ls'"

REM Kill a session when done
"C:\Program Files\PuTTY\plink.exe" -i "C:\Users\user\Documents\puttykey\mac.ppk" -P 8801 -batch gw@140.116.155.8 "bash -l -c 'tmux kill-session -t myjob'"
```

> Note: `tmux attach` requires an interactive terminal — it cannot be used through plink `-batch`.

## Transfer Files with pscp

`pscp.exe` (also from PuTTY) uses the same `.ppk` key for file transfer.

```powershell
# Upload local file to server
& "C:\Program Files\PuTTY\pscp.exe" -i "C:\Users\user\Documents\puttykey\mac.ppk" -P 8801 "C:\local\file.py" gw@140.116.155.8:/remote/path/

# Download file from server
& "C:\Program Files\PuTTY\pscp.exe" -i "C:\Users\user\Documents\puttykey\mac.ppk" -P 8801 gw@140.116.155.8:/remote/path/output.csv "C:\local\destination\"
```

## Why plink Instead of ssh

The standard `ssh` command on Windows requires OpenSSH format keys (`.pem` / no extension).
PuTTY `.ppk` keys can be converted with `puttygen -O private-openssh`, but `plink` skips that step entirely by reading `.ppk` directly.
