# 🚀 Project Setup Guide

This project uses **JAX** and **Flax**. Since these libraries have strict version dependencies, please follow the steps below to ensure everyone is working in the same environment and to avoid "ModuleNotFoundError" or "ImportError."

## 🛠 1. Environment Installation

Open your terminal and run the following commands in order:

### Create and Activate Conda Environment

```bash
conda env create -f environment.yml && conda activate gnn_project && python -m ipykernel install --user --name gnn_project --display-name "Python 3 (GNN_Project)"
```

---

## 💻 2. How to Run

1. Open your `.ipynb` notebook.
2. Look at the top-right corner to select the Kernel.
3. Choose **"Python 3 (GNN_Project)"**.
4. Run the first cell to verify:
```python
import flax
import jax
print(f"Flax version: {flax.__version__}")
print(f"JAX version: {jax.__version__}")

```

---

## ⚠️ Troubleshooting

If you see an `ImportError` regarding `get_default_device` or `pxla`, it means your JAX version was accidentally updated to a newer, incompatible version. To fix this:

1. Run: `pip install --force-reinstall jax==0.4.29 jaxlib==0.4.29`
2. **Restart the Kernel** in your Jupyter Notebook.

---

## 📢 Team Sync (Updating Packages)

If you install a new library (e.g., `seaborn` or `scipy`), please update the requirements so others can stay in sync:

* **To export:** `pip freeze > requirements.txt`
* **To update your own env from the team's list:** `pip install -r requirements.txt`

---
