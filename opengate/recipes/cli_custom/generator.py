from pathlib import Path
import logging

from opengate.utils.cli_utils import check_recipe_yaml

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

_command_run_dir = Path.cwd()


def write_dockerfile(content: str):
    dockerfile_dir = _command_run_dir / "training-plan-template"
    dockerfile_dir.mkdir(exist_ok=True)
    file_path = dockerfile_dir / "Dockerfile"
    file_path.write_text(content)


def read_template_file(template_path: Path) -> str:
    try:
        with template_path.open("r") as file:
            return file.read()
    except FileNotFoundError:
        raise Exception(f"Template file not found at: {template_path}")
    except Exception as e:
        raise Exception(f"Error reading template file: {e}")


def generate_local_dockerfile(command: str):
    try:
        training_plan_name = _command_run_dir.name
        check_recipe_yaml(command_run_dir=str(_command_run_dir), command=command)

        template_root_dir = Path(__file__).resolve().parents[3]

        template_path = template_root_dir / "opengate/recipes/cli_templates/dockerfile_template.txt"
        dockerfile_template = read_template_file(template_path)

        # Replace placeholders in the template
        dockerfile_content = dockerfile_template.format(
            template_root_dir=template_root_dir.name,
            training_plan_name=training_plan_name
        )

        write_dockerfile(dockerfile_content)
        _logger.info("Dockerfile successfully generated")
    except Exception as e:
        error_msg = f"Error occurred while generating Dockerfile: {e}"
        _logger.error(error_msg)
        raise Exception(error_msg)
