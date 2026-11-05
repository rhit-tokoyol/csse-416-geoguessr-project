# csse-416-geoguessr-project

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda --version

# Set identity
git config --global user.name "their-github-username"
git config --global user.email "their-github-email"

# Make an SSH key (press Enter to accept defaults)
ssh-keygen -t ed25519 -C "their-github-email"

# Print the public key, copy the whole line (starts with ssh-ed25519)
cat ~/.ssh/id_ed25519.pub

# Add it on GitHub: Settings > SSH and GPG keys > New SSH key > paste > save

# Test (type yes the first time)
ssh -T git@github.com

# Clone using the SSH URL
git clone git@github.com:<owner>/<repo>.git

# If already cloned over HTTPS, switch it
git remote set-url origin git@github.com:<owner>/<repo>.git

# Fix permissions if you get Permission denied (publickey)
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_ed25519

# Debug SSH
ssh -vT git@github.com

# If SSH is blocked, add this to ~/.ssh/config to use port 443
Host github.com
Hostname ssh.github.com
Port 443
User git

# Notes
# Compute nodes may have no internet, so do git operations on the login node or OnDemand, not in sbatch.
# Your friend must be a collaborator on the repo to push (repo Settings > Collaborators).
# Keep output/, logs/, and datasets out of git with a .gitignore.
