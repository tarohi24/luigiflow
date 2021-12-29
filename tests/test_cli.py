import subprocess


def test_cli_help():
    subprocess.run(["luigiflow", "--help"])
