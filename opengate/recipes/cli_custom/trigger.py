import os
import logging
import subprocess
import sys

from opengate.utils.cli_utils import check_recipe_yaml

logging.basicConfig(level=logging.DEBUG)
_logger = logging.getLogger(__name__)
_current_run_dur = os.getcwd()

def build_run_containers(command: str):
    check_recipe_yaml(command_run_dir=_current_run_dur, command=command)
    try:
        subprocess.run(
            ["docker", "compose", "up", "--build", "-d"],
            cwd=_current_run_dur,
            check=True,  # Raise an exception if the command fails
            text=True,  # Capture output as text
            stdout=sys.stdout,  # Stream stdout to the terminal
            stderr=sys.stderr  # Stream stderr to the terminal
        )
        _logger.info("Containers are built. Training is in progress...")

        # Fetch and display logs for the containers in real time
        _logger.info("Fetching logs for training-container...")
        with subprocess.Popen(
                ["docker", "logs", "-f", "training-container"],
                cwd=_current_run_dur,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,  # Line-buffered
                universal_newlines=True
        ) as process:
            for line in process.stdout:
                _logger.info(line.strip())  # Log each line in real time
            for line in process.stderr:
                _logger.error(line.strip())  # Log errors in real time
    except Exception as e:
        error_msg = f"Error occurred while building docker image: {e}"
        _logger.error(error_msg)
        raise Exception(error_msg)


def run_trainer():
    subprocess.run(
        ["docker", "compose", "run", "--rm", "training-container"],
        cwd=_current_run_dur,
        check=True,  # Raise an exception if the command fails
        text=True,  # Capture output as text
        stdout=sys.stdout,  # Stream stdout to the terminal
        stderr=sys.stderr  # Stream stderr to the terminal
    )
