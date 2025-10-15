# LeadScore AI - Lead Prioritization System

AI-powered lead prioritization system for B2B sales automation. Uses machine learning to identify and prioritize high-value prospects with 65.96% test accuracy.

## 📊 Quick Overview

- **Model Performance**: 65.96% test AUC (Random Forest)
- **Business Impact**: 67.8% of conversions captured in top 20% of leads
- **Revenue Multiplier**: 2.8x higher revenue for high-priority leads
- **Features**: 34+ engineered features with 5-dimensional scoring system

## 📋 Table of Contents

- [Documentation](#-documentation)
- [Key Visualizations](#-key-visualizations)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Model Performance](#-model-performance)
- [API Usage](#-api-usage)
- [Project Structure](#-project-structure)
- [Advanced Features](#-advanced-features)

## 📚 Documentation

### Technical Report
**[📄 Complete Technical Report](docs/LeadScore_AI_Technical_Report.pdf)**

The comprehensive technical report includes:
- Detailed methodology and feature engineering
- Statistical analysis and model comparison
- Business impact assessment
- Implementation guidelines
- Performance benchmarks and validation results

### Additional Documentation
- **API Documentation**: Available at `/docs` when server is running
- **Jupyter Notebooks**: Interactive analysis in [`notebooks/`](notebooks/) directory
- **LaTeX Source**: Technical report source in [`relatorio_leadscore_ai_melhorado.tex`](relatorio_leadscore_ai_melhorado.tex)

## 📈 Key Visualizations

### Model Performance Analysis
- **[Feature Importance](outputs/plots/model_evaluation/feature_importance.png)**: Top 10 most influential features
- **[ROC Curves](outputs/plots/model_evaluation/roc_curves.png)**: Model comparison across algorithms
- **[Precision-Recall Curves](outputs/plots/model_evaluation/precision_recall_curves.png)**: Performance trade-offs
- **[Model Comparison](outputs/plots/model_evaluation/model_performance_comparison.png)**: Algorithm benchmarking

### Business Intelligence Dashboards
- **[Segment Analysis](outputs/plots/business_dashboards/segment_analysis.png)**: Performance by industry segment
- **[Revenue Cohort Analysis](outputs/plots/business_dashboards/revenue_cohort_analysis.png)**: Revenue-based lead segmentation
- **[Temporal Analysis](outputs/plots/business_dashboards/temporal_analysis.png)**: Time-based conversion patterns

### Data Distribution Analysis
- **[Correlation Matrix](outputs/plots/correlations/correlation_matrix.png)**: Feature relationships
- **[Target Correlation](outputs/plots/correlations/target_correlation.png)**: Feature-target relationships
- **[Revenue Distribution](outputs/plots/distributions/faturamento_anual_milhoes_detailed.png)**: Revenue patterns
- **[Engagement Distribution](outputs/plots/distributions/reunioes_realizadas_detailed.png)**: Meeting engagement patterns

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/LucasTramonte/LeadScore-AI-Lead-Prioritization-System.git
   cd LeadScore-AI-Lead-Prioritization-System
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare data**:
   - Place lead data in `data/raw/data.xlsx`
   - Ensure data follows the expected schema (see [Data Format](#data-format))

## ⚡ Quick Start

### Train a Model
```bash
python train_model.py
# Or use CLI: python -m src.cli.main train --data data/raw/data.xlsx
```

### Score Leads
```bash
python test_scoring.py
# Or use CLI: python -m src.cli.main score --model model_name --input leads.xlsx --output results.xlsx
```

### Start API Server
```bash
python -m src.api.main
# API available at http://localhost:8000
# Documentation at http://localhost:8000/docs
```

## 📊 Model Performance

### Current Best Model: Random Forest
- **Test AUC**: 65.96%
- **Cross-validation AUC**: 69.36% ± 8.69%
- **Features**: 15 optimized engagement features
- **Training Date**: 2025-10-15

### Algorithm Comparison
| Algorithm | Test AUC | CV AUC | Status |
|-----------|----------|---------|---------|
| **Random Forest** | **65.96%** | **69.36% ± 8.69%** | **✅ Selected** |
| Gradient Boosting | 66.19% | 66.19% ± 4.73% | Available |
| CatBoost | 62.29% | 64.19% ± 8.76% | Available |
| Logistic Regression | 58.18% | 61.39% ± 10.75% | Baseline |

### Business Impact Metrics
- **Conversion Capture**: 67.8% of conversions in top 20% of leads
- **Efficiency Ratio**: 3.39x efficiency for top 20% leads
- **Revenue Impact**: 2.8x higher average revenue per conversion
- **Time Savings**: 40% reduction in lead qualification time

## 🔌 API Usage

### Endpoints
- `GET /health` - Health check
- `GET /models` - List available models
- `POST /score/lead` - Score a single lead
- `POST /score/batch` - Score multiple leads
- `GET /model/{model_name}/info` - Get model information

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

## 📁 Project Structure

```
LeadScore-AI-Lead-Prioritization-System/
├── 📚 docs/                         # Documentation
│   └── LeadScore_AI_Technical_Report.pdf
├── 🔧 src/                          # Source code
│   ├── api/                         # FastAPI web service
│   ├── cli/                         # Command line interface
│   ├── core/                        # Core configuration management
│   ├── data/                        # Data processing modules
│   ├── integrations/                # External integrations (Pipedrive, webhooks)
│   ├── model/                       # ML model training and persistence
│   ├── scoring/                     # Lead scoring functionality
│   └── services/                    # Business logic services
├── ⚙️ config/                       # Configuration files
├── 💾 data/                         # Data storage
├── 🤖 models/                       # Trained model artifacts
├── 📓 notebooks/                    # Jupyter notebooks for analysis
├── 📊 outputs/                      # Generated outputs and visualizations
│   ├── plots/                       # All visualization outputs
│   │   ├── business_dashboards/     # Business intelligence plots
│   │   ├── correlations/            # Correlation analysis
│   │   ├── distributions/           # Data distribution plots
│   │   └── model_evaluation/        # Model performance plots
│   └── tables/                      # Analysis results in tabular format
└── 📋 requirements.txt              # Python dependencies
```

## 🎯 Advanced Features

### 5-Dimensional Scoring System
1. **Engagement Score** (35%): Email interactions, meetings, content downloads
2. **Company Quality** (25%): Revenue, industry segment, export capability
3. **Decision Maker Access** (15%): Contact role and seniority level
4. **Lead Source Quality** (15%): Channel effectiveness and qualification
5. **Temporal Factor** (10%): Recency and urgency indicators

### Feature Engineering
- **Engagement Metrics**: Email open rates, response rates, meeting conversion
- **Company Intelligence**: Revenue normalization, industry benchmarking
- **Behavioral Patterns**: RFM analysis (Recency, Frequency, Monetary)
- **Temporal Features**: Time decay functions, urgency detection
- **Interaction Terms**: Cross-feature relationships for complex patterns

### Integration Capabilities
- **Pipedrive CRM**: Real-time lead synchronization
- **Webhook Support**: Automated scoring triggers
- **Batch Processing**: Large dataset handling
- **API Integration**: RESTful endpoints for external systems

## 🛠️ CLI Commands

```bash
# Model Management
python -m src.cli.main train --data data/raw/data.xlsx
python -m src.cli.main list
python -m src.cli.main info --model model_name

# Lead Scoring
python -m src.cli.main score --model model_name --input leads.xlsx --output results.xlsx
python -m src.cli.main evaluate --model model_name --data test_data.xlsx

# Data Operations
python -m src.cli.main validate --data data.xlsx
```

## 📋 Data Format

### Required Columns
- `segmento`: Industry segment
- `faturamento_anual_milhoes`: Annual revenue (millions)
- `numero_SKUs`: Number of SKUs
- `exporta`: Export status (0/1)
- `margem_media_setor`: Average sector margin
- `contact_role`: Contact role/position
- `lead_source`: Lead source
- `crm_stage`: CRM stage
- `emails_enviados`: Emails sent
- `emails_abertos`: Emails opened
- `emails_respondidos`: Emails responded
- `reunioes_realizadas`: Meetings held
- `download_whitepaper`: Whitepaper downloads (0/1)
- `demo_solicitada`: Demo requested (0/1)
- `problemas_reportados_precificacao`: Pricing problems reported (0/1)
- `urgencia_projeto`: Project urgency (0/1)
- `days_since_first_touch`: Days since first contact
- `converted`: Target variable (0/1) - for training only

## 🧪 Testing

```bash
python test_scoring.py                    # Basic functionality test
python test_advanced_features.py         # Advanced features test
python test_automation.py               # API automation test (requires server)
```

## 📈 Performance Monitoring

The system includes comprehensive performance tracking:
- **Model drift detection**: Automatic performance monitoring
- **Feature importance tracking**: Changes in feature relevance
- **Business metrics**: Conversion rates and revenue impact
- **Data quality checks**: Automated validation and alerts

## 🔄 Version History

- **v1.2.0** (Current): Enhanced documentation and visualization organization
- **v1.1.0**: Added advanced features and optimization
- **v1.0.0**: Initial release with core functionality

---

**LeadScore AI** - Transforming B2B sales through intelligent lead prioritization.

For detailed technical information, please refer to the [📄 Technical Report](docs/LeadScore_AI_Technical_Report.pdf).
