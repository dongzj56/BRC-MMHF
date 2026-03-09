
import os
import torch
import numpy as np
import huggingface_hub
from tabpfn import TabPFNClassifier

def main():
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        try:
            huggingface_hub.login(token=hf_token)
        except Exception:
            pass

    X_train = np.random.rand(50, 10)
    y_train = np.random.randint(0, 2, 50)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        classifier = TabPFNClassifier(device=device)
        classifier.fit(X_train, y_train)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model_instance = None
    if hasattr(classifier, 'model'):
        model_instance = classifier.model
    elif hasattr(classifier, 'classifier'):
        model_instance = classifier.classifier

    if model_instance is not None and isinstance(model_instance, torch.nn.Module):
        total_params = sum(p.numel() for p in model_instance.parameters())
        trainable_params = sum(p.numel() for p in model_instance.parameters() if p.requires_grad)
        
        print(f"Model: {type(model_instance).__name__}")
        print(f"Total Params:     {total_params:,}")
        print(f"Trainable Params: {trainable_params:,}")
        print(f"Size (approx):    {total_params * 4 / 1024 / 1024:.2f} MB")
    else:
        print("Could not locate internal PyTorch model.")

if __name__ == "__main__":
    main()
