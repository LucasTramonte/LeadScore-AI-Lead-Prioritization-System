# LeadScore AI - Lead Prioritization System

AI-powered lead prioritization system for B2B sales automation. Uses machine learning to identify and prioritize high-value prospects with 65.96% test accuracy.

## Quick Overview

- **Model Performance**: 65.96% test AUC (Random Forest)
- **Business Impact**: 67.8% of conversions captured in top 20% of leads
- **Revenue Multiplier**: 2.8x higher revenue for high-priority leads

## Table of Contents

- [Documentation](#documentation)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Usage](#api-usage)
- [Project Structure](#project-structure)

## Documentation

### Technical Report
**[Complete Technical Report](docs/LeadScore_AI_Technical_Report.pdf)**

### Key Visualizations
- **[Feature Importance](outputs/plots/model_evaluation/feature_importance.png)**
- **[Model Performance](outputs/plots/model_evaluation/model_performance_comparison.png)**
- **[Business Analysis](outputs/plots/business_dashboards/segment_analysis.png)**

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/LucasTramonte/LeadScore-AI-Lead-Prioritization-System.git
   cd LeadScore-AI-Lead-Prioritization-System
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Train a Model
```bash
python train_model.py
```

### Score Leads
```bash
python test_scoring.py
```

### Start API Server
```bash
python -m src.api.main
# API available at http://localhost:8000
# Documentation at http://localhost:8000/docs
```

## Model Performance

### Current Best Model: Random Forest
- **Test AUC**: 65.96%
- **Cross-validation AUC**: 69.36% ± 8.69%

### Algorithm Comparison
| Algorithm | Test AUC | Status |
|-----------|----------|---------|
| **Random Forest** | **65.96%** | **Selected** |
| Gradient Boosting | 66.19% | Available |
| CatBoost | 62.29% | Available |
| Logistic Regression | 58.18% | Baseline |

## API Usage

### Key Endpoints
- `GET /health` - Health check
- `POST /score/lead` - Score a single lead
- `POST /score/batch` - Score multiple leads

### Example Request
```python
import requests

lead_data = {
    "segmento": "Energia & Utilities",
    "faturamento_anual_milhoes": 50.5,
    "numero_SKUs": 120,
    "exporta": 1,
    "margem_media_setor": 15.0,
    "contact_role": "Diretor Financeiro (CFO)",
    "lead_source": "Evento Setorial",
    "crm_stage": "Qualificado Vendas",
    "emails_enviados": 5,
    "emails_abertos": 3,
    "emails_respondidos": 1,
    "reunioes_realizadas": 2,
    "download_whitepaper": 1,
    "demo_solicitada": 1,
    "problemas_reportados_precificacao": 1,
    "urgencia_projeto": 1,
    "days_since_first_touch": 15
}

response = requests.post("http://localhost:8000/score/lead", json=lead_data)
result = response.json()
print(f"Priority: {result['priority']}, Probability: {result['conversion_probability']:.1%}")
```

## Project Structure

```
LeadScore-AI-Lead-Prioritization-System/
├── docs/                         # Documentation
├── src/                          # Source code
│   ├── api/                      # FastAPI web service
│   ├── cli/                      # Command line interface
│   ├── data/                     # Data processing modules
│   ├── model/                    # ML model training
│   └── scoring/                  # Lead scoring functionality
├── config/                       # Configuration files
├── data/                         # Data storage
├── models/                       # Trained model artifacts
├── outputs/                      # Generated outputs and visualizations
└── requirements.txt              # Python dependencies
```

## CLI Commands

```bash
# Model Management
python -m src.cli.main train --data data/raw/data.xlsx
python -m src.cli.main list

# Lead Scoring
python -m src.cli.main score --model model_name --input leads.xlsx --output results.xlsx
```

## Testing

```bash
python test_scoring.py                    # Basic functionality test
python test_advanced_features.py         # Advanced features test
```

---

**LeadScore AI** - Transforming B2B sales through intelligent lead prioritization.

For detailed technical information, please refer to the [Technical Report](docs/LeadScore_AI_Technical_Report.pdf).
