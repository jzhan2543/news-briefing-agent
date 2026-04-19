"""
Package init for the news briefing agent.

Loads .env at package-import time so every submodule — nodes, tools,
runner — sees the API keys in os.environ regardless of which module is
imported first. This avoids the 'runner loads .env but web_search
imports earlier' ordering trap.
"""

from dotenv import load_dotenv

load_dotenv()