"""
Webhook handler for Pipedrive integration.
Processes incoming webhooks and triggers lead scoring automation.
"""

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import logging
import hashlib
import hmac
import json
from datetime import datetime
import asyncio
import os

from .pipedrive_client import PipedriveClient, PipedriveDataMapper
from .automation_engine import AutomationEngine
from ..api.main import score_lead_async

logger = logging.getLogger(__name__)

# Initialize webhook app
webhook_app = FastAPI(
    title="LeadScore AI Webhook Handler",
    description="Handles Pipedrive webhooks for automated lead scoring",
    version="1.0.0"
)

webhook_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for webhook payloads
class WebhookEvent(BaseModel):
    """Base webhook event model."""
    event: str = Field(..., description="Event type (e.g., 'create.person')")
    timestamp: datetime = Field(..., description="Event timestamp")
    user_id: int = Field(..., description="User ID who triggered the event")
    company_id: int = Field(..., description="Company ID")
    current: Dict[str, Any] = Field(..., description="Current object data")
    previous: Optional[Dict[str, Any]] = Field(None, description="Previous object data (for updates)")

class PipedriveWebhookPayload(BaseModel):
    """Pipedrive webhook payload model."""
    v: int = Field(..., description="Webhook version")
    matches_filters: Dict[str, Any] = Field(..., description="Filter matches")
    meta: Dict[str, Any] = Field(..., description="Metadata")
    current: Dict[str, Any] = Field(..., description="Current object data")
    previous: Optional[Dict[str, Any]] = Field(None, description="Previous object data")
    event: str = Field(..., description="Event type")
    retry: int = Field(0, description="Retry count")

# Global automation engine instance
automation_engine = None

def get_automation_engine() -> AutomationEngine:
    """Get or create automation engine instance."""
    global automation_engine
    if automation_engine is None:
        # Initialize with environment variables
        pipedrive_token = os.getenv('PIPEDRIVE_API_TOKEN')
        pipedrive_domain = os.getenv('PIPEDRIVE_DOMAIN', 'amazon4')
        
        if not pipedrive_token:
            logger.warning("PIPEDRIVE_API_TOKEN not set, automation features will be limited")
            automation_engine = AutomationEngine(None, pipedrive_domain)
        else:
            automation_engine = AutomationEngine(pipedrive_token, pipedrive_domain)
    
    return automation_engine

def verify_webhook_signature(payload: bytes, signature: str, secret: str) -> bool:
    """
    Verify webhook signature for security.
    
    Args:
        payload: Raw webhook payload
        signature: Webhook signature from headers
        secret: Webhook secret
        
    Returns:
        True if signature is valid
    """
    if not secret:
        logger.warning("No webhook secret configured, skipping signature verification")
        return True
    
    try:
        expected_signature = hmac.new(
            secret.encode('utf-8'),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(f"sha256={expected_signature}", signature)
    except Exception as e:
        logger.error(f"Error verifying webhook signature: {str(e)}")
        return False

async def process_lead_scoring(event_data: Dict[str, Any], event_type: str) -> Dict[str, Any]:
    """
    Process lead scoring for webhook event.
    
    Args:
        event_data: Event data from webhook
        event_type: Type of event (create.person, create.deal, etc.)
        
    Returns:
        Scoring results
    """
    try:
        automation_engine = get_automation_engine()
        
        # Extract object ID and type
        object_id = event_data.get('id')
        object_type = event_type.split('.')[1]  # e.g., 'person' from 'create.person'
        
        if not object_id:
            raise ValueError("No object ID found in event data")
        
        # Get additional data from Pipedrive if needed
        additional_data = await automation_engine.get_enriched_lead_data(object_id, object_type)
        
        # Map Pipedrive data to LeadScore format
        mapped_data = PipedriveDataMapper.map_pipedrive_to_leadscore(
            event_data, 
            additional_data.get('activities', [])
        )
        
        # Score the lead
        scoring_result = await score_lead_async(mapped_data)
        
        # Trigger automation based on score
        automation_result = await automation_engine.execute_automation(
            object_id, object_type, scoring_result, event_data
        )
        
        return {
            'object_id': object_id,
            'object_type': object_type,
            'scoring_result': scoring_result,
            'automation_result': automation_result,
            'processed_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing lead scoring: {str(e)}")
        raise

# Webhook endpoints
@webhook_app.post("/webhooks/pipedrive")
async def handle_pipedrive_webhook(
    request: Request,
    background_tasks: BackgroundTasks
):
    """
    Handle incoming Pipedrive webhooks.
    
    Args:
        request: FastAPI request object
        background_tasks: Background task handler
        
    Returns:
        Webhook processing response
    """
    try:
        # Get raw payload and headers
        payload = await request.body()
        signature = request.headers.get('X-Pipedrive-Signature', '')
        webhook_secret = os.getenv('PIPEDRIVE_WEBHOOK_SECRET', '')
        
        # Verify signature if secret is configured
        if webhook_secret and not verify_webhook_signature(payload, signature, webhook_secret):
            logger.warning("Invalid webhook signature")
            raise HTTPException(status_code=401, detail="Invalid signature")
        
        # Parse JSON payload
        try:
            webhook_data = json.loads(payload.decode('utf-8'))
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON payload: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid JSON payload")
        
        # Validate webhook payload
        try:
            webhook_payload = PipedriveWebhookPayload(**webhook_data)
        except Exception as e:
            logger.error(f"Invalid webhook payload structure: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid payload structure")
        
        # Check if this is an event we care about
        supported_events = [
            'create.person', 'create.deal', 'create.lead',
            'change.person', 'change.deal', 'change.lead'
        ]
        
        if webhook_payload.event not in supported_events:
            logger.info(f"Ignoring unsupported event: {webhook_payload.event}")
            return {"status": "ignored", "reason": "unsupported_event"}
        
        # Process in background to avoid timeout
        background_tasks.add_task(
            process_webhook_background,
            webhook_payload.dict(),
            webhook_payload.event
        )
        
        return {
            "status": "accepted",
            "event": webhook_payload.event,
            "object_id": webhook_payload.current.get('id'),
            "processed_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in webhook handler: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

async def process_webhook_background(webhook_data: Dict[str, Any], event_type: str):
    """
    Process webhook in background task.
    
    Args:
        webhook_data: Webhook payload data
        event_type: Type of webhook event
    """
    try:
        logger.info(f"Processing webhook event: {event_type}")
        
        # Process lead scoring
        result = await process_lead_scoring(webhook_data['current'], event_type)
        
        logger.info(f"Successfully processed webhook: {result['object_id']} - {result['scoring_result']['priority']}")
        
    except Exception as e:
        logger.error(f"Error processing webhook in background: {str(e)}")

@webhook_app.get("/webhooks/health")
async def webhook_health_check():
    """Health check for webhook service."""
    try:
        automation_engine = get_automation_engine()
        pipedrive_status = "connected" if automation_engine.pipedrive_client else "disconnected"
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "pipedrive_status": pipedrive_status,
            "supported_events": [
                "create.person", "create.deal", "create.lead",
                "change.person", "change.deal", "change.lead"
            ]
        }
    except Exception as e:
        logger.error(f"Webhook health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@webhook_app.post("/webhooks/test")
async def test_webhook_processing(test_data: Dict[str, Any]):
    """
    Test endpoint for webhook processing.
    
    Args:
        test_data: Test webhook data
        
    Returns:
        Processing results
    """
    try:
        event_type = test_data.get('event', 'create.person')
        event_data = test_data.get('current', test_data)
        
        result = await process_lead_scoring(event_data, event_type)
        
        return {
            "status": "success",
            "test_result": result
        }
        
    except Exception as e:
        logger.error(f"Test webhook processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Webhook management endpoints
@webhook_app.post("/webhooks/setup")
async def setup_pipedrive_webhooks(
    webhook_url: str,
    events: Optional[List[str]] = None
):
    """
    Set up Pipedrive webhooks.
    
    Args:
        webhook_url: URL to receive webhooks
        events: List of events to subscribe to
        
    Returns:
        Setup results
    """
    try:
        automation_engine = get_automation_engine()
        
        if not automation_engine.pipedrive_client:
            raise HTTPException(status_code=503, detail="Pipedrive client not configured")
        
        # Default events if none specified
        if not events:
            events = ['create.person', 'create.deal', 'create.lead']
        
        results = []
        for event in events:
            try:
                action, obj = event.split('.')
                result = automation_engine.pipedrive_client.create_webhook(
                    subscription_url=webhook_url,
                    event_action=action,
                    event_object=obj,
                    name=f"LeadScore AI - {event}"
                )
                results.append({
                    "event": event,
                    "status": "created",
                    "webhook_id": result['data']['id']
                })
            except Exception as e:
                results.append({
                    "event": event,
                    "status": "failed",
                    "error": str(e)
                })
        
        return {
            "status": "completed",
            "webhook_url": webhook_url,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error setting up webhooks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@webhook_app.get("/webhooks/list")
async def list_pipedrive_webhooks():
    """List all configured Pipedrive webhooks."""
    try:
        automation_engine = get_automation_engine()
        
        if not automation_engine.pipedrive_client:
            raise HTTPException(status_code=503, detail="Pipedrive client not configured")
        
        webhooks = automation_engine.pipedrive_client.list_webhooks()
        
        return {
            "status": "success",
            "webhooks": webhooks
        }
        
    except Exception as e:
        logger.error(f"Error listing webhooks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@webhook_app.delete("/webhooks/{webhook_id}")
async def delete_pipedrive_webhook(webhook_id: int):
    """Delete a Pipedrive webhook."""
    try:
        automation_engine = get_automation_engine()
        
        if not automation_engine.pipedrive_client:
            raise HTTPException(status_code=503, detail="Pipedrive client not configured")
        
        result = automation_engine.pipedrive_client.delete_webhook(webhook_id)
        
        return {
            "status": "deleted",
            "webhook_id": webhook_id,
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Error deleting webhook: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Startup event
@webhook_app.on_event("startup")
async def startup_webhook_handler():
    """Initialize webhook handler."""
    logger.info("Starting LeadScore AI Webhook Handler...")
    try:
        # Initialize automation engine
        get_automation_engine()
        logger.info("Webhook handler startup completed successfully")
    except Exception as e:
        logger.error(f"Webhook handler startup failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.integrations.webhook_handler:webhook_app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
