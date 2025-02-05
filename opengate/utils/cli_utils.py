import os

def check_recipe_yaml(command_run_dir: str, command: str):
    recipe_path = os.path.join(command_run_dir, "recipe.yaml")
    if not os.path.exists(recipe_path):
        raise Exception(f"{command} command must be run in the directory where recipe.yaml file exists")