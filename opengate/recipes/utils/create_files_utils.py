import os
from opengate.recipes.steps.ingest import IngestStep
from opengate.recipes.steps.split import SplitStep
from opengate.recipes.steps.transform import TransformStep
from opengate.recipes.steps.train import TrainStep
from opengate.recipes.steps.evaluate import EvaluateStep
from opengate.recipes.steps.register import RegisterStep
from opengate.recipes.steps.predict import PredictStep
from opengate.recipes.steps.ingest import IngestScoringStep

def check_and_create_file(file_path: str):
    if not os.path.exists(path=file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # Create an empty file
        with open(file_path, 'w') as file:
            pass

def check_and_creat_folder(folder_path: str):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def ingest(mlflow_recipe_dir: str, base_dir: str):
    ingest_conf = os.path.join(mlflow_recipe_dir, "ingest/conf.yaml")
    check_and_create_file(ingest_conf)
    output_dir = os.path.join(mlflow_recipe_dir, "ingest/outputs")
    check_and_creat_folder(output_dir)
    ingest_step = IngestStep.from_step_config_path(step_config_path=ingest_conf, recipe_root=base_dir)
    ingest_step.run(output_directory=output_dir)
    print("Ingest step completed")

def split(mlflow_recipe_dir: str, base_dir: str):
    split_conf = os.path.join(mlflow_recipe_dir, "split/conf.yaml")
    check_and_create_file(split_conf)
    output_dir = os.path.join(mlflow_recipe_dir, "split/outputs")
    check_and_creat_folder(output_dir)
    split_step = SplitStep.from_step_config_path(step_config_path=split_conf, recipe_root=base_dir)
    split_step.run(output_directory=output_dir)
    print("Split step completed")

def transform(mlflow_recipe_dir: str, base_dir: str):
    transform_conf = os.path.join(mlflow_recipe_dir, "transform/conf.yaml")
    check_and_create_file(transform_conf)
    output_dir = os.path.join(mlflow_recipe_dir, "transform/outputs")
    check_and_creat_folder(output_dir)
    transform_step = TransformStep.from_step_config_path(step_config_path=transform_conf, recipe_root=base_dir)
    transform_step.run(output_directory=output_dir)
    print("Transform step completed")

def train(mlflow_recipe_dir: str, base_dir: str):
    train_conf = os.path.join(mlflow_recipe_dir, "train/conf.yaml")
    check_and_create_file(train_conf)
    output_dir = os.path.join(mlflow_recipe_dir, "train/outputs")
    check_and_creat_folder(output_dir)
    train_step = TrainStep.from_step_config_path(step_config_path=train_conf, recipe_root=base_dir)
    train_step.run(output_directory=output_dir)
    print("Train step completed")

def evaluate(mlflow_recipe_dir: str, base_dir: str):
    evaluate_conf = os.path.join(mlflow_recipe_dir, "evaluate/conf.yaml")
    check_and_create_file(evaluate_conf)
    output_dir = os.path.join(mlflow_recipe_dir, "evaluate/outputs")
    check_and_creat_folder(output_dir)
    evaluate_step = EvaluateStep.from_step_config_path(step_config_path=evaluate_conf, recipe_root=base_dir)
    evaluate_step.run(output_directory=output_dir)
    print("Evaluate step completed")

def register(mlflow_recipe_dir: str, base_dir: str):
    register_conf = os.path.join(mlflow_recipe_dir, "register/conf.yaml")
    check_and_create_file(register_conf)
    output_dir = os.path.join(mlflow_recipe_dir, "register/outputs")
    check_and_creat_folder(output_dir)
    register_step = RegisterStep.from_step_config_path(step_config_path=register_conf, recipe_root=base_dir)
    register_step.run(output_directory=output_dir)
    print("Register step completed")

def ingest_scoring(mlflow_recipe_dir: str, base_dir: str):
    ingest_scoring_conf = os.path.join(mlflow_recipe_dir, "ingest_scoring/conf.yaml")
    check_and_create_file(ingest_scoring_conf)
    output_dir = os.path.join(mlflow_recipe_dir, "ingest_scoring/outputs")
    check_and_creat_folder(output_dir)
    ingest_scoring_step = IngestScoringStep.from_step_config_path(step_config_path=ingest_scoring_conf, recipe_root=base_dir)
    ingest_scoring_step.run(output_directory=output_dir)
    print("Ingest scoring step completed")

def predict(mlflow_recipe_dir: str, base_dir: str):
    predict_conf = os.path.join(mlflow_recipe_dir, "predict/conf.yaml")
    check_and_create_file(predict_conf)
    output_dir = os.path.join(mlflow_recipe_dir, "predict/outputs")
    check_and_creat_folder(output_dir)
    predict_step = PredictStep.from_step_config_path(step_config_path=predict_conf, recipe_root=base_dir)
    predict_step.run(output_directory=output_dir)
    print("Predict step completed")

# Clean function to remove generated output files
def clean(mlflow_recipe_dir: str):
    output_dirs = [
        os.path.join(mlflow_recipe_dir, "steps/split/outputs/train.parquet"),
        os.path.join(mlflow_recipe_dir, "steps/split/outputs/validation.parquet"),
        os.path.join(mlflow_recipe_dir, "steps/split/outputs/test.parquet"),
        os.path.join(mlflow_recipe_dir, "steps/transform/outputs/transformer.pkl"),
        os.path.join(mlflow_recipe_dir, "steps/transform/outputs/transformed_training_data.parquet"),
        os.path.join(mlflow_recipe_dir, "steps/transform/outputs/transformed_validation_data.parquet"),
        os.path.join(mlflow_recipe_dir, "steps/train/outputs/model"),
        os.path.join(mlflow_recipe_dir, "steps/train/outputs/run_id"),
        os.path.join(mlflow_recipe_dir, "steps/evaluate/outputs/model_validation_status"),
        os.path.join(mlflow_recipe_dir, "steps/predict/outputs/scored.parquet")
    ]
    for dir_path in output_dirs:
        if os.path.exists(dir_path):
            print(f"Removing directory: {dir_path}")
            os.system(f"rm -rf {dir_path}")
    print("Clean completed")

def start_creating(mlflow_recipe_dir: str, project_base_dir: str):
    mlflow_recipe_steps_dir = mlflow_recipe_dir + "/steps"
    ingest(mlflow_recipe_steps_dir, project_base_dir)
    split(mlflow_recipe_steps_dir, project_base_dir)
    transform(mlflow_recipe_steps_dir, project_base_dir)
    train(mlflow_recipe_steps_dir, project_base_dir)
    evaluate(mlflow_recipe_steps_dir, project_base_dir)
    register(mlflow_recipe_steps_dir, project_base_dir)
    ingest_scoring(mlflow_recipe_steps_dir, project_base_dir)
    predict(mlflow_recipe_steps_dir, project_base_dir)
    clean(mlflow_recipe_dir)
