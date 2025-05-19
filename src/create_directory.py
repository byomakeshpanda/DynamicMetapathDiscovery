import os

def create_project_structure(base_dir):
    structure = {
        "data": ["raw", "processed", "visualization"],
        "src": {
            "data": ["__init__.py", "data_loader.py", "preprocessing.py"],
            "models": ["__init__.py", "base_model.py", "relation_attention.py", "metapath_attention.py", "dynamic_metapath_gnn.py"],
            "utils": ["__init__.py", "graph_utils.py", "evaluation_metrics.py"],
            "visualization": ["__init__.py", "attention_heatmaps.py", "metapath_visualizer.py"],
            "explainability": ["__init__.py", "counterfactual_analysis.py", "metapath_interpreter.py"]
        },
        "experiments": ["configs", "logs", "results"],
        "notebooks": ["exploratory_data_analysis.ipynb", "model_evaluation.ipynb"],
        "tests": ["test_data_processing.py", "test_models.py", "test_explainability.py"],
        "docs": ["project_specification.md", "architecture_design.md", "final_report.md"],
        "": ["requirements.txt", "setup.py", "README.md", ".gitignore"]
    }

    def create_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)

    def create_file(path):
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        if not os.path.exists(path):
            with open(path, 'w') as f:
                pass

    for key, value in structure.items():
        if isinstance(value, list):
            for item in value:
                if "." in item:  # It's a file
                    create_file(os.path.join(base_dir, key, item))
                else:  # It's a directory
                    create_dir(os.path.join(base_dir, key, item))
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                sub_dir = os.path.join(base_dir, key, sub_key)
                create_dir(sub_dir)
                for item in sub_value:
                    if "." in item:  # It's a file
                        create_file(os.path.join(sub_dir, item))
                    else:  # It's a directory
                        create_dir(os.path.join(sub_dir, item))

# Example usage
base_directory = "dynamic_metapath_discovery"
create_project_structure(base_directory)
print(f"Project structure created under {base_directory}")
