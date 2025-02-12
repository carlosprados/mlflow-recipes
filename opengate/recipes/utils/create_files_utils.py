import os
import logging
from typing import Dict
import shutil

from opengate.recipes.steps.ingest import IngestStep
from opengate.recipes.steps.split import SplitStep
from opengate.recipes.steps.split_anomaly import SplitAnomalyStep
from opengate.recipes.steps.transform import TransformStep
from opengate.recipes.steps.train import TrainStep
from opengate.recipes.steps.evaluate import EvaluateStep
from opengate.recipes.steps.register import RegisterStep
from opengate.recipes.steps.predict import PredictStep
from opengate.recipes.steps.ingest import IngestScoringStep
from opengate.recipes.steps.train_anomaly import TrainAnomalyStep
from opengate.recipes.steps.evaluate_anomaly import EvaluateAnomalyStep
from opengate.recipes.steps.transform_anomaly import TransformAnomalyStep
from opengate.recipes.task_enum import MLTask

_logger = logging.getLogger(__name__)

def check_and_create_file(file_path: str):
    if not os.path.exists(path=file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # Create an empty file
        with open(file_path, 'w'):
            pass


def check_and_create_folder(folder_path: str):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def get_container_data(ingest_config: Dict[str, str]):
    container_loc = ingest_config.get("container_location")
    if container_loc is not None or container_loc != "":
        filename = container_loc.split("/")[-1]
        data_target_loc = os.path.join(os.getcwd(), "data", filename)
        shutil.move(container_loc, data_target_loc)


class CreateMlflowFiles:
    def __init__(self, mlflow_recipe_dir: str, project_base_dir: str, target: str, template: str):
        self.mlflow_recipe_steps_dir = "/".join([mlflow_recipe_dir, "steps"])
        self.project_base_dir = project_base_dir
        self.target = target
        self.template = template

    def ingest(self):
        ingest_conf = os.path.join(self.mlflow_recipe_steps_dir, "ingest/conf.yaml")
        check_and_create_file(ingest_conf)
        output_dir = os.path.join(self.mlflow_recipe_steps_dir, "ingest/outputs")
        check_and_create_folder(output_dir)
        ingest_step = IngestStep.from_step_config_path(step_config_path=ingest_conf, recipe_root=self.project_base_dir)
        get_container_data(ingest_step.step_config)
        ingest_step.run(output_directory=output_dir)
        _logger.info("Ingest step completed")

    def split(self):
        split_conf = os.path.join(self.mlflow_recipe_steps_dir, "split/conf.yaml")
        check_and_create_file(split_conf)
        output_dir = os.path.join(self.mlflow_recipe_steps_dir, "split/outputs")
        check_and_create_folder(output_dir)
        step_class = {
            MLTask.ANOMALY.value: SplitAnomalyStep,
        }.get(self.template, SplitStep)
        split_step = step_class.from_step_config_path(step_config_path=split_conf, recipe_root=self.project_base_dir)
        split_step.run(output_directory=output_dir)
        _logger.info("Split step completed")

    def transform(self):
        transform_conf = os.path.join(self.mlflow_recipe_steps_dir, "transform/conf.yaml")
        check_and_create_file(transform_conf)
        output_dir = os.path.join(self.mlflow_recipe_steps_dir, "transform/outputs")
        check_and_create_folder(output_dir)
        step_class = {
            MLTask.ANOMALY.value: TransformAnomalyStep,
        }.get(self.template, TransformStep)
        transform_step = step_class.from_step_config_path(step_config_path=transform_conf,
                                                          recipe_root=self.project_base_dir)
        transform_step.run(output_directory=output_dir)
        _logger.info("Transform step completed")

    def train(self):
        train_conf = os.path.join(self.mlflow_recipe_steps_dir, "train/conf.yaml")
        check_and_create_file(train_conf)
        output_dir = os.path.join(self.mlflow_recipe_steps_dir, "train/outputs")
        check_and_create_folder(output_dir)

        step_class = {
            MLTask.ANOMALY.value: TrainAnomalyStep,
        }.get(self.template, TrainStep)
        train_step = step_class.from_step_config_path(step_config_path=train_conf, recipe_root=self.project_base_dir)

        train_step.run(output_directory=output_dir)
        _logger.info("Train step completed")

    def evaluate(self):
        evaluate_conf = os.path.join(self.mlflow_recipe_steps_dir, "evaluate/conf.yaml")
        check_and_create_file(evaluate_conf)
        output_dir = os.path.join(self.mlflow_recipe_steps_dir, "evaluate/outputs")
        check_and_create_folder(output_dir)

        step_class = {
            MLTask.ANOMALY.value: EvaluateAnomalyStep,
        }.get(self.template, EvaluateStep)
        evaluate_step = step_class.from_step_config_path(step_config_path=evaluate_conf, recipe_root=self.project_base_dir)

        evaluate_step.run(output_directory=output_dir)
        _logger.info("Evaluate step completed")

    def register(self):
        register_conf = os.path.join(self.mlflow_recipe_steps_dir, "register/conf.yaml")
        check_and_create_file(register_conf)
        output_dir = os.path.join(self.mlflow_recipe_steps_dir, "register/outputs")
        check_and_create_folder(output_dir)
        run_id_file = os.path.join(output_dir, "run_id")
        check_and_create_file(run_id_file)
        register_step = RegisterStep.from_step_config_path(step_config_path=register_conf, recipe_root=self.project_base_dir)
        register_step.run(output_directory=output_dir)
        _logger.info("Register step completed")

    def ingest_scoring(self):
        ingest_scoring_conf = os.path.join(self.mlflow_recipe_steps_dir, "ingest_scoring/conf.yaml")
        output_dir = os.path.join(self.mlflow_recipe_steps_dir, "ingest_scoring/outputs")
        scoring_output_file = os.path.join(output_dir, "scoring-dataset.parquet")
        message = "Ingest scoring step skipped: Output file already exists."

        # Check if the output file already exists
        if not os.path.exists(scoring_output_file):
            check_and_create_file(ingest_scoring_conf)
            check_and_create_folder(output_dir)

            ingest_scoring_step = IngestScoringStep.from_step_config_path(
                step_config_path=ingest_scoring_conf,
                recipe_root=self.project_base_dir
            )
            ingest_scoring_step.run(output_directory=output_dir)
            message = "Ingest scoring step completed"
        _logger.info(message)

    def predict(self):
        predict_conf = os.path.join(self.mlflow_recipe_steps_dir, "predict/conf.yaml")
        check_and_create_file(predict_conf)
        output_dir = os.path.join(self.mlflow_recipe_steps_dir, "predict/outputs")
        check_and_create_folder(output_dir)
        predict_step = PredictStep.from_step_config_path(step_config_path=predict_conf, recipe_root=self.project_base_dir)
        predict_step.run(output_directory=output_dir)
        _logger.info("Predict step completed")

    # Clean function to remove generated output files
    def clean(self):
        output_dirs = [
            os.path.join(self.mlflow_recipe_steps_dir, "steps/split/outputs/train.parquet"),
            os.path.join(self.mlflow_recipe_steps_dir, "steps/split/outputs/validation.parquet"),
            os.path.join(self.mlflow_recipe_steps_dir, "steps/split/outputs/test.parquet"),
            os.path.join(self.mlflow_recipe_steps_dir, "steps/transform/outputs/transformer.pkl"),
            os.path.join(self.mlflow_recipe_steps_dir, "steps/transform/outputs/transformed_training_data.parquet"),
            os.path.join(self.mlflow_recipe_steps_dir, "steps/transform/outputs/transformed_validation_data.parquet"),
            os.path.join(self.mlflow_recipe_steps_dir, "steps/train/outputs/model"),
            os.path.join(self.mlflow_recipe_steps_dir, "steps/train/outputs/run_id"),
            os.path.join(self.mlflow_recipe_steps_dir, "steps/evaluate/outputs/model_validation_status"),
            os.path.join(self.mlflow_recipe_steps_dir, "steps/predict/outputs/scored.parquet")
        ]
        for dir_path in output_dirs:
            if os.path.exists(dir_path):
                _logger.info(f"Removing directory: {dir_path}")
                os.system(f"rm -rf {dir_path}")
        _logger.info("Clean completed")

    def start_creating(self):
        match self.target:
            case "ingest":
                self.ingest()
            case "split":
                self.split()
            case "transform":
                self.transform()
            case "train":
                self.train()
            case "evaluate":
                self.evaluate()
            case "register":
                self.register()
            case "ingest_scoring":
                self.ingest_scoring()
            case "predict":
                self.predict()
            case "clean":
                self.clean()

