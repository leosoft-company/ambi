"""Session-wide test setup — load .env so smoke tests see API keys."""

from ambi.env import load_env

load_env()
