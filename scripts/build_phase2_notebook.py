import json
from pathlib import Path

def create_phase2_notebook():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Cross-Dataset Robustness & In-Domain Generalization Benchmark (Phase 1 & Phase 2)\n",
                "\n",
                "This notebook executes the complete cross-border and in-domain evaluation protocol across **Germany (GTSRB)**, **Belgium (BelgiumTSC)**, and **China (Chinese TSRD)**:\n",
                "\n",
                "1. **Phase 1 Robustness Benchmark:** Probing frozen German-trained representations under 15 Hendrycks & Dietterich corruptions across 5 severities (75 conditions) under **Static BN** vs. **Test-Time BN Adaptation** on foreign roadways.\n",
                "2. **Phase 2 In-Domain Training:** Training black-box **Baseline ResNet-18** and **Frequency-Aware CBM** directly on Belgian (62 classes) and Chinese (58 classes) roads with decoupled No-Bias-Decay SGD and Cosine Annealing.\n",
                "3. **Phase 2 In-Domain Robustness Benchmark:** Testing in-domain models across all 75 corruption conditions to evaluate whether in-domain models resolve the Covariate Shift Paradox."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ==============================================================================\n",
                "# Step 1: Environment Setup, Drive Mount & GPU Verification\n",
                "# ==============================================================================\n",
                "from google.colab import drive\n",
                "import torch\n",
                "import os\n",
                "\n",
                "drive.mount('/content/drive')\n",
                "print('PyTorch Version:', torch.__version__)\n",
                "print('CUDA Available: ', torch.cuda.is_available())\n",
                "if torch.cuda.is_available():\n",
                "    print('Active GPU:     ', torch.cuda.get_device_name(0))\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ==============================================================================\n",
                "# Step 2: Unpack Phase 2 Code & Standardized Datasets\n",
                "# ==============================================================================\n",
                "import os\n",
                "\n",
                "# Unpack code package from Google Drive or session storage\n",
                "drive_pkg = '/content/drive/MyDrive/phase2_colab_package.zip'\n",
                "local_pkg = '/content/phase2_colab_package.zip'\n",
                "\n",
                "if os.path.exists(drive_pkg):\n",
                "    print('Copying', drive_pkg, 'to local runtime...')\n",
                "    !cp /content/drive/MyDrive/phase2_colab_package.zip /content/\n",
                "    !unzip -q -o /content/phase2_colab_package.zip -d /content/ML_Project\n",
                "elif os.path.exists(local_pkg):\n",
                "    print('Unpacking', local_pkg, '...')\n",
                "    !unzip -q -o /content/phase2_colab_package.zip -d /content/ML_Project\n",
                "else:\n",
                "    print('WARNING: phase2_colab_package.zip not found! Please upload it to MyDrive or /content/')\n",
                "\n",
                "%cd /content/ML_Project\n",
                "!python -c \"import torch; from src.models.cbm import ConceptBottleneckModel; print('Environment & Pre-Rendered Datasets Verified!')\"\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ==============================================================================\n",
                "# Step 3: Phase 1 Cross-Dataset Robustness Benchmark (75 Corruptions)\n",
                "# ==============================================================================\n",
                "# Evaluates German-trained Baseline and CBM models on Belgium and China across\n",
                "# 15 corruptions x 5 severities under both Static BN and Test-Time Adapted BN.\n",
                "\n",
                "!python scripts/evaluate_cross_dataset_robustness.py \\\n",
                "    --phase 1 \\\n",
                "    --dataset all \\\n",
                "    --batch-size 128 \\\n",
                "    --output-dir reports/cross_dataset_robustness\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ==============================================================================\n",
                "# Step 4: Phase 2 In-Domain Training (Belgium [62 classes] & China [58 classes])\n",
                "# ==============================================================================\n",
                "# Trains all four candidate models with Decoupled No-Bias-Decay SGD, Cosine Annealing & AMP (~10 mins on T4 GPU):\n",
                "\n",
                "# 1. Belgium Baseline ResNet-18 (62 classes)\n",
                "!python scripts/run_phase2_training.py --dataset belgium --model-type baseline --epochs 30 --batch-size 128 --lr 0.05\n",
                "\n",
                "# 2. Belgium Frequency-Aware CBM (62 classes)\n",
                "!python scripts/run_phase2_training.py --dataset belgium --model-type freq_cbm --epochs 30 --batch-size 128 --lr 0.05\n",
                "\n",
                "# 3. China Baseline ResNet-18 (58 classes)\n",
                "!python scripts/run_phase2_training.py --dataset china --model-type baseline --epochs 30 --batch-size 128 --lr 0.05\n",
                "\n",
                "# 4. China Frequency-Aware CBM (58 classes)\n",
                "!python scripts/run_phase2_training.py --dataset china --model-type freq_cbm --epochs 30 --batch-size 128 --lr 0.05\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ==============================================================================\n",
                "# Step 5: Phase 2 In-Domain Robustness Benchmark (75 Corruptions)\n",
                "# ==============================================================================\n",
                "# Evaluates all four newly trained in-domain models across 15 corruptions x 5 severities\n",
                "# under both Static BN and Test-Time Adapted BN.\n",
                "\n",
                "!python scripts/evaluate_cross_dataset_robustness.py \\\n",
                "    --phase 2 \\\n",
                "    --dataset all \\\n",
                "    --batch-size 128 \\\n",
                "    --output-dir reports/cross_dataset_robustness\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ==============================================================================\n",
                "# Step 6: Consolidate Metrics & Display Master Comparative Table\n",
                "# ==============================================================================\n",
                "import glob\n",
                "import json\n",
                "import pandas as pd\n",
                "\n",
                "records = []\n",
                "for path in sorted(glob.glob('reports/cross_dataset_robustness/*.json')):\n",
                "    with open(path) as f:\n",
                "        d = json.load(f)\n",
                "    records.append({\n",
                "        'Report': path.split('/')[-1],\n",
                "        'Dataset': d.get('dataset', '').upper(),\n",
                "        'Model': d.get('model', d.get('model_type', '')),\n",
                "        'Clean Acc (%)': round(d.get('clean_acc', d.get('clean_top1_acc', 0.0)) * 100, 2),\n",
                "        'Static mCA (%)': round(d.get('static_mCA', 0.0) * 100, 2),\n",
                "        'Static RPD (%)': round(d.get('static_RPD', 0.0), 2),\n",
                "        'Adapt mCA (%)': round(d.get('adapt_mCA', 0.0) * 100, 2),\n",
                "        'Adapt RPD (%)': round(d.get('adapt_RPD', 0.0), 2),\n",
                "    })\n",
                "\n",
                "df_summary = pd.DataFrame(records)\n",
                "display(df_summary)\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ==============================================================================\n",
                "# Step 7: Automatic Backup to Google Drive & Download Archive\n",
                "# ==============================================================================\n",
                "!mkdir -p \"/content/drive/MyDrive/ML_Project_Phase2_Results\"\n",
                "!cp -r reports/cross_dataset_robustness \"/content/drive/MyDrive/ML_Project_Phase2_Results/\"\n",
                "!cp -r checkpoints \"/content/drive/MyDrive/ML_Project_Phase2_Results/\"\n",
                "\n",
                "# Also create a downloadable zip in Colab root\n",
                "!zip -q -r /content/phase2_results_complete.zip reports/cross_dataset_robustness checkpoints\n",
                "print('\\nPhase 2 Complete! Checkpoints and reports synchronized to Google Drive: MyDrive/ML_Project_Phase2_Results/')\n"
            ]
        }
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "provenance": [],
                "gpuType": "T4"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    nb_path = Path("notebooks/phase2_cross_dataset_robustness_colab.ipynb")
    nb_path.parent.mkdir(parents=True, exist_ok=True)
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print("Successfully created notebook at:", nb_path)

if __name__ == "__main__":
    create_phase2_notebook()

