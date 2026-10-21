# Connecting to HUIT Open OnDemand with SSH

This guide is for macOS or Linux. You’ll need access to the HUIT Open OnDemand
(OOD) web dashboard and a connection to the **Harvard VPN**.

## 1. Check for an existing SSH key

In a terminal **on your own computer**, run:

```bash
ls ~/.ssh/id_ed25519.pub
```

If the file exists, continue to step 2. If it doesn’t, create a key:

```bash
ssh-keygen -t ed25519 -C "your_email@[g|college].harvard.edu"
```

Press **Enter** to accept the default location, then follow the prompts to choose
a passphrase. If prompted to overwrite an existing key, **stop and keep the
existing key**.

This creates two files:

- `~/.ssh/id_ed25519`: your **private key**. Keep it on your computer; never share it.
- `~/.ssh/id_ed25519.pub`: your **public key**, which you’ll copy to OOD.

## 2. Copy your public key

**On your computer**, run:

```bash
cat ~/.ssh/id_ed25519.pub
```

Copy the entire line, beginning with `ssh-ed25519`.

On macOS, you can alternatively copy it directly to your clipboard:

```bash
pbcopy < ~/.ssh/id_ed25519.pub
```

This command produces no output. Run it on your Mac, not inside the OOD terminal.

## 3. Add your public key to OOD

Open the **terminal app in the OOD web dashboard**. Run these commands there:

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
touch ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
nano ~/.ssh/authorized_keys
```

Paste your public key on its **own new line**. Preserve any existing keys.

Save and exit:

1. Press **Ctrl+O**, then **Enter** to save.
2. Press **Ctrl+X** to exit.

You only need to do this once per computer/key. Copy only the `.pub` file;
your private key stays on your computer.

## 4. Find your OOD username

In the **OOD web terminal**, run:

```bash
whoami
```

The result is your Harvard NetID. Use this username for SSH; it may differ from
your email address or other cluster usernames.

## 5. Connect from your computer

Connect to the **Harvard VPN**, then run this in your **local terminal**,
replacing `YOUR_NETID`:

```bash
ssh YOUR_NETID@login.ood.huit.harvard.edu
```

On your first connection, SSH will ask whether to trust the server. Confirm that
you entered the correct hostname, then type `yes`. If your key has a passphrase,
enter it when prompted.

SSH automatically checks the default key location. To explicitly select the key, use:

```bash
ssh -i ~/.ssh/id_ed25519 YOUR_NETID@login.ood.huit.harvard.edu
```

To disconnect and return to your local terminal:

```bash
exit
```

## Troubleshooting

- **Connection hangs or times out:** Check your Harvard VPN connection.
- **Permission denied:** Verify your NetID, confirm the correct public key is in
  OOD’s `~/.ssh/authorized_keys`, and repeat the permission commands in step 3.
  Each key must occupy a single line.
- **Host identification changed:** Check with course staff before removing the
  old host entry.

If you get **Too many authentication failures**, tell SSH to use only your selected key:

```bash
ssh -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 YOUR_NETID@login.ood.huit.harvard.edu
```

## Once you’re connected

You’re on a **shared login node**. Submit compute workloads through Slurm using
`sbatch` or `srun`; don’t run heavy computations directly on the login node.

Adapted from HUIT’s *Accessing HUIT OOD via SSH* setup guide.
