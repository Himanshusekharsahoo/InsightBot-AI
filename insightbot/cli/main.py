"""InsightBot AI CLI Entry Point.

Private AI Research & Knowledge Agent for Teams.
"""

import os
import sys

# Set program name for argument parsing
os.environ.setdefault('INSIGHTBOT_CLI_PROG', 'insightbot')

from browser_use.skill_cli.main import main as _engine_cli_main


def main() -> int:
	"""Main entry point for insightbot CLI."""
	if len(sys.argv) > 1 and sys.argv[1] in ('preflight', 'check-config', '--preflight'):
		from insightbot.config.preflight import run_preflight_cli

		return run_preflight_cli(sys.argv[2:])
	return _engine_cli_main()


if __name__ == '__main__':
	sys.exit(main())
