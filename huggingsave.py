from huggingface_hub import HfApi, login
import os

def upload_to_hf(data_path, model_path, token, repo_id):
    # Login avec le token
    login(token)
    api = HfApi()
    
    # Upload données
    print(f"Upload des données depuis {data_path}...")
    api.upload_file(
        path_or_fileobj=data_path,
        path_in_repo="market_data.npy",
        repo_id=repo_id,
        repo_type="dataset"
    )
    
    # Upload modèle seulement s'il existe
    if os.path.exists(model_path):
        print(f"Upload du modèle depuis {model_path}...")
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo="model.pt",
            repo_id=repo_id,
            repo_type="dataset"
        )
    else:
        print(f"Modèle non trouvé à {model_path}, skip de l'upload du modèle")
    
    print(f"Upload terminé: https://huggingface.co/{repo_id}")

# Usage:
if __name__ == "__main__":
    upload_to_hf(
        "market_data_20241108_144746.npy",
        "best_model.pt",
        "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", 
        "Zual/st"
    )