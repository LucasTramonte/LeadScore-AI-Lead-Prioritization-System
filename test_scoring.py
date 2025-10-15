#!/usr/bin/env python3
"""
Aprix Lead Scoring System - Test Script

Simple test script to verify lead scoring functionality.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.scoring.lead_scorer import LeadScorer

def main():
    """Test lead scoring functionality"""
    print("=== Aprix Lead Scoring System - Test ===\n")
    
    # Sample lead data
    sample_lead = {
        'segmento': 'Energia & Utilities',
        'faturamento_anual_milhoes': 50.5,
        'numero_SKUs': 120,
        'exporta': 1,
        'margem_media_setor': 15.0,
        'contact_role': 'Diretor Financeiro (CFO)',
        'lead_source': 'Evento Setorial',
        'crm_stage': 'Qualificado Vendas',
        'emails_enviados': 5,
        'emails_abertos': 3,
        'emails_respondidos': 1,
        'reunioes_realizadas': 2,
        'download_whitepaper': 1,
        'demo_solicitada': 1,
        'problemas_reportados_precificacao': 1,
        'urgencia_projeto': 1,
        'days_since_first_touch': 15
    }
    
    try:
        # Load the latest model
        print("Loading trained model...")
        model_name = "leadscore_random_forest_20251015_132536"
        scorer = LeadScorer(model_name)
        
        print("Sample Lead Data:")
        for key, value in sample_lead.items():
            print(f"  {key}: {value}")
        
        print("\nScoring the lead...")
        result = scorer.score_lead(sample_lead)
        
        print("\n=== SCORING RESULTS ===")
        print(f"Conversion Probability: {result['conversion_probability']:.1%}")
        print(f"Priority: {result['priority']}")
        print(f"Confidence Score: {result.get('confidence_score', 'N/A')}")
        
        print("\n=== INTERPRETATION ===")
        if result['priority'] == 'High':
            print("🔥 HIGH PRIORITY LEAD!")
            print("   → Immediate follow-up recommended")
            print("   → Assign to senior sales rep")
            print("   → Schedule demo/meeting ASAP")
        elif result['priority'] == 'Medium':
            print("⚡ MEDIUM PRIORITY LEAD")
            print("   → Follow-up within 24-48 hours")
            print("   → Send targeted content")
            print("   → Monitor engagement")
        else:
            print("📋 LOW PRIORITY LEAD")
            print("   → Add to nurturing campaign")
            print("   → Monitor for engagement changes")
        
        print("\n=== SUCCESS! ===")
        print("The AI model successfully scored your lead!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Make sure you have trained a model first by running: python train_model.py")

if __name__ == "__main__":
    main()
