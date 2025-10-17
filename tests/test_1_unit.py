import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import train_model
import numpy as np

def test_model_shape():
    """Unit test for the ML model training function."""
    print("\nRunning unit test for train_model()...")

    X, y, model = train_model()
    print(f"Feature matrix shape: {X.shape}")
    print(f"Unique target classes: {np.unique(y)}")
    print(f"Model type: {type(model).__name__}")

    # Assertions
    assert X.shape[1] > 0, "Feature matrix has no columns!"
    assert len(np.unique(y)) > 1, "Target labels are not diverse!"

    print("✅ Unit test passed: model trained successfully with valid data and labels.")