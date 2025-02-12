from pathlib import Path
import logging
from mlflow import version

from opengate.utils.cli_utils import check_recipe_yaml

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

_command_run_dir = Path.cwd()
_dockerfile_template_relative_path = "opengate/recipes/cli_templates/dockerfile-template.txt"
_docker_compose_template_relative_path = "opengate/recipes/cli_templates/docker-compose-template.txt"


def write_dockerfile(content: str, filepath_to_save: str):
    generated_folders_dir = Path(_command_run_dir / filepath_to_save)
    generated_folders_dir.parent.mkdir(parents=True, exist_ok=True)
    generated_folders_dir.write_text(content)


def read_template_file(template_path: Path) -> str:
    try:
        with template_path.open("r") as file:
            return file.read()
    except FileNotFoundError:
        raise Exception(f"Template file not found at: {template_path}")
    except Exception as e:
        raise Exception(f"Error reading template file: {e}")


def generate_local_container_files(command: str):
    try:
        training_plan_name = _command_run_dir.name
        check_recipe_yaml(command_run_dir=str(_command_run_dir), command=command)

        template_root_dir = Path(__file__).resolve().parents[3]

        dockerfile_template_path = template_root_dir / _dockerfile_template_relative_path
        dockerfile_template = read_template_file(dockerfile_template_path)

        # Replace placeholders in the template
        dockerfile_content = dockerfile_template.format(
            template_root_dir=template_root_dir.name,
            training_plan_name=training_plan_name
        )

        docker_compose_template_path = template_root_dir / _docker_compose_template_relative_path
        docker_compose_template = read_template_file(docker_compose_template_path)

        docker_compose_content = docker_compose_template.format(
            context=_command_run_dir.parent,
            plan_folder_name=training_plan_name,
            mlflow_version=version.VERSION
        )

        write_dockerfile(content=dockerfile_content, filepath_to_save="generated/Dockerfile")
        _logger.info("Dockerfile successfully generated")
        write_dockerfile(content=docker_compose_content, filepath_to_save="docker-compose.yaml")
        _logger.info("Docker compose successfully generated")
    except Exception as e:
        error_msg = f"Error occurred while generating Dockerfile: {e}"
        _logger.error(error_msg)
        raise Exception(error_msg)
