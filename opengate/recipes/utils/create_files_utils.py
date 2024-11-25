import os
from opengate.recipes.steps.ingest import IngestStep
from opengate.recipes.steps.split import SplitStep
from opengate.recipes.steps.transform import TransformStep
from opengate.recipes.steps.train import TrainStep
from opengate.recipes.steps.evaluate import EvaluateStep
from opengate.recipes.steps.register import RegisterStep
from opengate.recipes.steps.predict import PredictStep
from opengate.recipes.steps.ingest import IngestScoringStep
from opengate.recipes.steps.train_isolation_forest import TrainIsolationForest

def check_and_create_file(file_path: str):
    if not os.path.exists(path=file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # Create an empty file
        with open(file_path, 'w') as file:
            pass


def check_and_create_folder(folder_path: str):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


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
        ingest_step.run(output_directory=output_dir)
        print("Ingest step completed")

    def split(self):
        split_conf = os.path.join(self.mlflow_recipe_steps_dir, "split/conf.yaml")
        check_and_create_file(split_conf)
        output_dir = os.path.join(self.mlflow_recipe_steps_dir, "split/outputs")
        check_and_create_folder(output_dir)
        split_step = SplitStep.from_step_config_path(step_config_path=split_conf, recipe_root=self.project_base_dir)
        split_step.run(output_directory=output_dir)
        print("Split step completed")

    def transform(self):
        transform_conf = os.path.join(self.mlflow_recipe_steps_dir, "transform/conf.yaml")
        check_and_create_file(transform_conf)
        output_dir = os.path.join(self.mlflow_recipe_steps_dir, "transform/outputs")
        check_and_create_folder(output_dir)
        transform_step = TransformStep.from_step_config_path(step_config_path=transform_conf, recipe_root=self.project_base_dir)
        transform_step.run(output_directory=output_dir)
        print("Transform step completed")

    def train(self):
        train_conf = os.path.join(self.mlflow_recipe_steps_dir, "train/conf.yaml")
        check_and_create_file(train_conf)
        output_dir = os.path.join(self.mlflow_recipe_steps_dir, "train/outputs")
        check_and_create_folder(output_dir)
        match self.template:
            case "anomaly/v1":
                train_step = TrainIsolationForest.from_step_config_path(step_config_path=train_conf,
                                                                        recipe_root=self.project_base_dir)
            case _:
                train_step = TrainStep.from_step_config_path(step_config_path=train_conf, recipe_root=self.project_base_dir)
        train_step.run(output_directory=output_dir)
        print("Train step completed")

    def evaluate(self):
        evaluate_conf = os.path.join(self.mlflow_recipe_steps_dir, "evaluate/conf.yaml")
        check_and_create_file(evaluate_conf)
        output_dir = os.path.join(self.mlflow_recipe_steps_dir, "evaluate/outputs")
        check_and_create_folder(output_dir)
        evaluate_step = EvaluateStep.from_step_config_path(step_config_path=evaluate_conf, recipe_root=self.project_base_dir)
        evaluate_step.run(output_directory=output_dir)
        print("Evaluate step completed")

    def register(self):
        register_conf = os.path.join(self.mlflow_recipe_steps_dir, "register/conf.yaml")
        check_and_create_file(register_conf)
        output_dir = os.path.join(self.mlflow_recipe_steps_dir, "register/outputs")
        check_and_create_folder(output_dir)
        run_id_file = os.path.join(output_dir, "run_id")
        check_and_create_file(run_id_file)
        register_step = RegisterStep.from_step_config_path(step_config_path=register_conf, recipe_root=self.project_base_dir)
        register_step.run(output_directory=output_dir)
        print("Register step completed")

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
        print(message)

    def predict(self):
        predict_conf = os.path.join(self.mlflow_recipe_steps_dir, "predict/conf.yaml")
        check_and_create_file(predict_conf)
        output_dir = os.path.join(self.mlflow_recipe_steps_dir, "predict/outputs")
        check_and_create_folder(output_dir)
        predict_step = PredictStep.from_step_config_path(step_config_path=predict_conf, recipe_root=self.project_base_dir)
        predict_step.run(output_directory=output_dir)
        print("Predict step completed")

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
                print(f"Removing directory: {dir_path}")
                os.system(f"rm -rf {dir_path}")
        print("Clean completed")

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

