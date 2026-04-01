import os

# Default save directory (following XDG specs)
SAVE_DIR = os.path.join(os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache')), 'hy3dgen', 'archeon')
os.makedirs(SAVE_DIR, exist_ok=True)
