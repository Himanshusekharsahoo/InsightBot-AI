"""InsightBot AI Configuration Package."""

from insightbot.config.settings import CONFIG, InsightBotConfig, Settings, get_settings, load_insightbot_config


def check_configuration(*args, **kwargs):
	from insightbot.config.preflight import check_configuration as _check

	return _check(*args, **kwargs)


def run_preflight_cli(*args, **kwargs):
	from insightbot.config.preflight import run_preflight_cli as _run

	return _run(*args, **kwargs)


__all__ = [
	'InsightBotConfig',
	'CONFIG',
	'load_insightbot_config',
	'Settings',
	'get_settings',
	'check_configuration',
	'run_preflight_cli',
]
