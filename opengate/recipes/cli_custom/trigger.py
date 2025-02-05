import os
import logging
import subprocess

from opengate.utils.cli_utils import check_recipe_yaml

logging.basicConfig(level=logging.DEBUG)
_logger = logging.getLogger(__name__)
_current_run_dur = os.getcwd()
os.environ['PATH'] = '/usr/bin:' + os.environ.get('PATH', '')

def build_docker(command: str):
    check_recipe_yaml(command_run_dir=_current_run_dur, command=command)
    dockerfile_folder_name = "training-plan-template"
    docker_folder_path = os.path.join(_current_run_dur, dockerfile_folder_name)
    dockerfile_name = "Dockerfile"
    training_plan_folder_name = _current_run_dur.split("/")[-1]
    container_name = "-".join([training_plan_folder_name, "container"])
    image_name = "-".join([training_plan_folder_name, "image"])

    dockerfile_path = os.path.join(_current_run_dur, dockerfile_folder_name, dockerfile_name)
    try:
        subprocess.run(
            ["docker", "build", "-t", image_name, "-f", dockerfile_path, docker_folder_path],
            check=True
        )

        subprocess.run(
            ["docker", "run", "--name", container_name, image_name],
            check=True
        )
        _logger.info(
            f"Docker imaged successfully generated with container name: {container_name} and image name {image_name}")
    except Exception as e:
        error_msg = f"Error occurred while building docker image {e}"
        _logger.error(error_msg)
        raise Exception(error_msg)

def run_docker():
    ...