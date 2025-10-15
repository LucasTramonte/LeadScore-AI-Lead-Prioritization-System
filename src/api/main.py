"""
FastAPI application for LeadScore AI REST API.
Provides endpoints for lead scoring and model management.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
import logging
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import uvicorn

from ..scoring.lead_scorer import LeadScorer
from ..model.model_persistence import ModelPersistence

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for caching
_default_scorer = None
_model_cache = {}
_app_start_time = datetime.now()

def get_default_scorer() -> LeadScorer:
    """Get or create default scorer instance."""
    global _default_scorer
    if _default_scorer is None:
        try:
            # Get the latest model
            persistence = ModelPersistence()
            models_df = persistence.list_saved_models()
            if len(models_df) == 0:
                raise HTTPException(status_code=503, detail="No trained models available")
            
            latest_model = models_df.iloc[0]['model_name']
            _default_scorer = LeadScorer(latest_model)
            logger.info(f"Loaded default scorer with model: {latest_model}")
        except Exception as e:
            logger.error(f"Failed to load default scorer: {str(e)}")
            raise HTTPException(status_code=503, detail=f"Failed to load scoring model: {str(e)}")
    
    return _default_scorer

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup
    logger.info("Starting LeadScore AI API...")
    try:
        # Pre-load default scorer
        get_default_scorer()
        logger.info("API startup completed successfully")
    except Exception as e:
        logger.error(f"API startup failed: {str(e)}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down LeadScore AI API...")
    executor.shutdown(wait=True)

# Initialize FastAPI app
app = FastAPI(
    title="LeadScore AI API",
    description="REST API for AI-powered lead scoring and prioritization",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=4)

# Pydantic models for request/response
class LeadData(BaseModel):
    """Lead data model for scoring requests."""
    segmento: str = Field(..., description="Industrial segment")
    faturamento_anual_milhoes: float = Field(..., description="Annual revenue in millions")
    numero_SKUs: int = Field(..., description="Number of SKUs")
    margem_media_setor: float = Field(..., description="Average sector margin")
    exporta: int = Field(..., description="Export status (0/1)")
    contact_role: str = Field(..., description="Contact person role")
    lead_source: str = Field(..., description="Lead source")
    crm_stage: str = Field(..., description="CRM pipeline stage")
    emails_enviados: int = Field(..., description="Emails sent")
    emails_abertos: int = Field(..., description="Emails opened")
    emails_respondidos: int = Field(..., description="Emails responded")
    reunioes_realizadas: int = Field(..., description="Meetings held")
    download_whitepaper: int = Field(..., description="Whitepaper downloads (0/1)")
    demo_solicitada: int = Field(..., description="Demo requested (0/1)")
    problemas_reportados_precificacao: int = Field(..., description="Pricing problems reported (0/1)")
    urgencia_projeto: int = Field(..., description="Project urgency (0/1)")
    days_since_first_touch: int = Field(..., description="Days since first contact")

class LeadScoreResponse(BaseModel):
    """Response model for lead scoring."""
    lead_id: Optional[str] = None
    conversion_probability: float = Field(..., description="Conversion probability (0-1)")
    priority: str = Field(..., description="Priority level (High/Medium/Low)")
    confidence: float = Field(..., description="Model confidence (0-1)")
    scores: Dict[str, float] = Field(..., description="Individual feature scores")
    model_info: Dict[str, Any] = Field(..., description="Model information")
    scored_at: datetime = Field(default_factory=datetime.now)

class BatchLeadRequest(BaseModel):
    """Request model for batch lead scoring."""
    leads: List[LeadData] = Field(..., description="List of leads to score")
    model_name: Optional[str] = Field(None, description="Specific model to use")
    include_details: bool = Field(True, description="Include detailed scoring information")

class BatchLeadResponse(BaseModel):
    """Response model for batch lead scoring."""
    results: List[LeadScoreResponse] = Field(..., description="Scoring results")
    summary: Dict[str, Any] = Field(..., description="Batch processing summary")
    processed_at: datetime = Field(default_factory=datetime.now)

class ModelInfo(BaseModel):
    """Model information response."""
    model_name: str
    model_type: str
    version: str
    training_date: datetime
    performance_metrics: Dict[str, float]
    thresholds: Dict[str, float]
    feature_requirements: Dict[str, List[str]]

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = "1.0.0"
    models_available: int
    uptime_seconds: float

def get_scorer(model_name: Optional[str] = None) -> LeadScorer:
    """Get scorer instance for specific model."""
    if model_name is None:
        return get_default_scorer()
    
    if model_name not in _model_cache:
        try:
            _model_cache[model_name] = LeadScorer(model_name)
            logger.info(f"Loaded scorer for model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")
    
    return _model_cache[model_name]

async def score_lead_async(lead_data: Dict[str, Any], model_name: Optional[str] = None) -> Dict[str, Any]:
    """Asynchronously score a single lead."""
    loop = asyncio.get_event_loop()
    scorer = get_scorer(model_name)
    
    def _score():
        return scorer.score_lead(lead_data)
    
    return await loop.run_in_executor(executor, _score)

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "LeadScore AI API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Check if models are available
        persistence = ModelPersistence()
        models_df = persistence.list_saved_models()
        models_count = len(models_df)
        
        # Calculate uptime
        uptime = (datetime.now() - _app_start_time).total_seconds()
        
        return HealthResponse(
            status="healthy" if models_count > 0 else "degraded",
            models_available=models_count,
            uptime_seconds=uptime
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            models_available=0,
            uptime_seconds=(datetime.now() - _app_start_time).total_seconds()
        )

@app.post("/score/lead", response_model=LeadScoreResponse)
async def score_single_lead(
    lead_data: LeadData,
    model_name: Optional[str] = None
):
    """Score a single lead and return priority classification."""
    try:
        # Convert Pydantic model to dict
        lead_dict = lead_data.dict()
        
        # Score the lead
        result = await score_lead_async(lead_dict, model_name)
        
        # Get model info
        scorer = get_scorer(model_name)
        model_info = scorer.get_model_info()
        
        return LeadScoreResponse(
            conversion_probability=result['conversion_probability'],
            priority=result['priority'],
            confidence=result['confidence_score'],
            scores={
                'company_quality_score': result.get('company_quality_score', 0.0),
                'engagement_score': result.get('engagement_score', 0.0),
                'lead_quality_score': result.get('lead_quality_score', 0.0)
            },
            model_info={
                'model_name': model_info['model_name'],
                'model_type': model_info['model_type'],
                'version': model_info['version']
            }
        )
        
    except Exception as e:
        logger.error(f"Error scoring lead: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")

@app.post("/score/batch", response_model=BatchLeadResponse)
async def score_batch_leads(request: BatchLeadRequest):
    """Score multiple leads in batch."""
    try:
        # Convert leads to list of dicts
        leads_data = [lead.dict() for lead in request.leads]
        
        # Score leads asynchronously
        tasks = [score_lead_async(lead_data, request.model_name) for lead_data in leads_data]
        results = await asyncio.gather(*tasks)
        
        # Convert results to response format
        lead_responses = []
        for i, result in enumerate(results):
            scorer = get_scorer(request.model_name)
            model_info = scorer.get_model_info()
            
            lead_responses.append(LeadScoreResponse(
                lead_id=f"lead_{i+1}",
                conversion_probability=result['conversion_probability'],
                priority=result['priority'],
                confidence=result['confidence_score'],
                scores={
                    'company_quality_score': result.get('company_quality_score', 0.0),
                    'engagement_score': result.get('engagement_score', 0.0),
                    'lead_quality_score': result.get('lead_quality_score', 0.0)
                },
                model_info={
                    'model_name': model_info['model_name'],
                    'model_type': model_info['model_type'],
                    'version': model_info['version']
                }
            ))
        
        # Generate summary
        priorities = [r.priority for r in lead_responses]
        summary = {
            'total_leads': len(lead_responses),
            'high_priority': priorities.count('High'),
            'medium_priority': priorities.count('Medium'),
            'low_priority': priorities.count('Low'),
            'average_probability': sum(r.conversion_probability for r in lead_responses) / len(lead_responses),
            'model_used': request.model_name or 'default'
        }
        
        return BatchLeadResponse(
            results=lead_responses,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Error in batch scoring: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch scoring failed: {str(e)}")

@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all available models."""
    try:
        persistence = ModelPersistence()
        models_df = persistence.list_saved_models()
        
        model_list = []
        for _, model_row in models_df.iterrows():
            try:
                scorer = LeadScorer(model_row['model_name'])
                model_info = scorer.get_model_info()
                
                model_list.append(ModelInfo(
                    model_name=model_info['model_name'],
                    model_type=model_info['model_type'],
                    version=model_info['version'],
                    training_date=datetime.fromisoformat(model_info['training_date']),
                    performance_metrics={
                        'test_auc': model_row.get('test_auc', 0.0)
                    },
                    thresholds=model_info['thresholds'],
                    feature_requirements=scorer.get_feature_requirements()['required_features']
                ))
            except Exception as e:
                logger.warning(f"Could not load model info for {model_row['model_name']}: {str(e)}")
                continue
        
        return model_list
        
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

@app.get("/models/{model_name}", response_model=ModelInfo)
async def get_model_info(model_name: str):
    """Get detailed information about a specific model."""
    try:
        scorer = LeadScorer(model_name)
        model_info = scorer.get_model_info()
        
        return ModelInfo(
            model_name=model_info['model_name'],
            model_type=model_info['model_type'],
            version=model_info['version'],
            training_date=datetime.fromisoformat(model_info['training_date']),
            performance_metrics=model_info.get('model_comparison', {}).get(model_info['model_type'], {}),
            thresholds=model_info['thresholds'],
            feature_requirements=scorer.get_feature_requirements()['required_features']
        )
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
