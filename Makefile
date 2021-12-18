pr := poetry run
fmttarget := luigiflow tests

mypy:
	$(pr) mypy --show-error-codes luigiflow

format-diff:
	$(pr) isort --diff $(fmttarget)
	$(pr) black --diff $(fmttarget)

format:
	$(pr) isort $(fmttarget)
	$(pr) black $(fmttarget)
