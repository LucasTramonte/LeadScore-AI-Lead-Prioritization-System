"""
Pipedrive API client for LeadScore AI integration.
Handles authentication, data retrieval, and updates.
"""

import requests
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class PipedriveClient:
    """
    Client for interacting with Pipedrive API.
    """
    
    def __init__(self, api_token: str, company_domain: str):
        """
        Initialize Pipedrive client.
        
        Args:
            api_token: Pipedrive API token
            company_domain: Company domain (e.g., 'amazon4')
        """
        self.api_token = api_token
        self.company_domain = company_domain
        self.base_url = f"https://{company_domain}.pipedrive.com/api/v1"
        self.session = requests.Session()
        self.session.params = {'api_token': api_token}
        
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make HTTP request to Pipedrive API.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint
            **kwargs: Additional request parameters
            
        Returns:
            API response data
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            
            data = response.json()
            if not data.get('success', False):
                raise Exception(f"API request failed: {data.get('error', 'Unknown error')}")
            
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Pipedrive API request failed: {str(e)}")
            raise Exception(f"API request failed: {str(e)}")
    
    def get_person(self, person_id: int) -> Dict[str, Any]:
        """
        Get person details by ID.
        
        Args:
            person_id: Person ID
            
        Returns:
            Person data
        """
        return self._make_request('GET', f'/persons/{person_id}')
    
    def get_organization(self, org_id: int) -> Dict[str, Any]:
        """
        Get organization details by ID.
        
        Args:
            org_id: Organization ID
            
        Returns:
            Organization data
        """
        return self._make_request('GET', f'/organizations/{org_id}')
    
    def get_deal(self, deal_id: int) -> Dict[str, Any]:
        """
        Get deal details by ID.
        
        Args:
            deal_id: Deal ID
            
        Returns:
            Deal data
        """
        return self._make_request('GET', f'/deals/{deal_id}')
    
    def get_lead(self, lead_id: str) -> Dict[str, Any]:
        """
        Get lead details by ID.
        
        Args:
            lead_id: Lead ID
            
        Returns:
            Lead data
        """
        return self._make_request('GET', f'/leads/{lead_id}')
    
    def update_person(self, person_id: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update person with new data.
        
        Args:
            person_id: Person ID
            data: Update data
            
        Returns:
            Updated person data
        """
        return self._make_request('PUT', f'/persons/{person_id}', json=data)
    
    def update_deal(self, deal_id: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update deal with new data.
        
        Args:
            deal_id: Deal ID
            data: Update data
            
        Returns:
            Updated deal data
        """
        return self._make_request('PUT', f'/deals/{deal_id}', json=data)
    
    def update_lead(self, lead_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update lead with new data.
        
        Args:
            lead_id: Lead ID
            data: Update data
            
        Returns:
            Updated lead data
        """
        return self._make_request('PATCH', f'/leads/{lead_id}', json=data)
    
    def create_activity(self, activity_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new activity.
        
        Args:
            activity_data: Activity data
            
        Returns:
            Created activity data
        """
        return self._make_request('POST', '/activities', json=activity_data)
    
    def create_note(self, note_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new note.
        
        Args:
            note_data: Note data
            
        Returns:
            Created note data
        """
        return self._make_request('POST', '/notes', json=note_data)
    
    def get_person_activities(self, person_id: int) -> List[Dict[str, Any]]:
        """
        Get activities for a person.
        
        Args:
            person_id: Person ID
            
        Returns:
            List of activities
        """
        response = self._make_request('GET', f'/persons/{person_id}/activities')
        return response.get('data', [])
    
    def get_deal_activities(self, deal_id: int) -> List[Dict[str, Any]]:
        """
        Get activities for a deal.
        
        Args:
            deal_id: Deal ID
            
        Returns:
            List of activities
        """
        response = self._make_request('GET', f'/deals/{deal_id}/activities')
        return response.get('data', [])
    
    def create_webhook(self, subscription_url: str, event_action: str, event_object: str, 
                      name: str, user_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Create a new webhook.
        
        Args:
            subscription_url: URL to receive webhook notifications
            event_action: Event action (create, change, delete, *)
            event_object: Event object (person, deal, lead, *)
            name: Webhook name
            user_id: User ID for authorization
            
        Returns:
            Created webhook data
        """
        webhook_data = {
            'subscription_url': subscription_url,
            'event_action': event_action,
            'event_object': event_object,
            'name': name,
            'version': '2.0'
        }
        
        if user_id:
            webhook_data['user_id'] = user_id
        
        return self._make_request('POST', '/webhooks', json=webhook_data)
    
    def list_webhooks(self) -> List[Dict[str, Any]]:
        """
        List all webhooks.
        
        Returns:
            List of webhooks
        """
        response = self._make_request('GET', '/webhooks')
        return response.get('data', [])
    
    def delete_webhook(self, webhook_id: int) -> Dict[str, Any]:
        """
        Delete a webhook.
        
        Args:
            webhook_id: Webhook ID
            
        Returns:
            Deletion response
        """
        return self._make_request('DELETE', f'/webhooks/{webhook_id}')


class PipedriveDataMapper:
    """
    Maps Pipedrive data to LeadScore AI format.
    """
    
    # Mapping configurations
    SEGMENT_MAPPING = {
        'Food & Beverages': 'Alimentos & Bebidas',
        'Chemicals': 'Químicos & Plásticos',
        'Metallurgy': 'Metalurgia',
        'Machinery & Equipment': 'Máquinas & Equipamentos',
        'Construction': 'Construção',
        'Energy & Utilities': 'Energia & Utilities',
        'Consumer Goods': 'Bens de Consumo'
    }
    
    CONTACT_ROLE_MAPPING = {
        'Pricing Analyst': 'Analista de Preços',
        'CFO': 'Diretor Financeiro (CFO)',
        'Operations Director': 'Diretor de Operações',
        'Financial Manager': 'Gerente Financeiro',
        'Cost Coordinator': 'Coordenador de Custos',
        'Commercial Manager': 'Gerente Comercial',
        'Pricing Analyst': 'Analista de Precificação'
    }
    
    LEAD_SOURCE_MAPPING = {
        'Industry Event': 'Evento Setorial',
        'Client Referral': 'Indicação de Cliente',
        'Website': 'Inbound (Site)',
        'Cold Outreach': 'Prospecção Ativa',
        'Technical Content': 'Conteúdo Técnico'
    }
    
    CRM_STAGE_MAPPING = {
        'New': 'Novo',
        'Qualified': 'Qualificação',
        'Proposal': 'Proposta',
        'Negotiation': 'Negociação'
    }
    
    @classmethod
    def map_pipedrive_to_leadscore(cls, pipedrive_data: Dict[str, Any], 
                                  activities: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Map Pipedrive data to LeadScore format.
        
        Args:
            pipedrive_data: Raw Pipedrive data
            activities: List of activities for engagement metrics
            
        Returns:
            Mapped data for LeadScore
        """
        # Extract organization data
        org_data = pipedrive_data.get('org_id', {}) if isinstance(pipedrive_data.get('org_id'), dict) else {}
        
        # Calculate engagement metrics from activities
        engagement_metrics = cls._calculate_engagement_metrics(activities or [])
        
        # Map the data
        mapped_data = {
            # Company information
            'segmento': cls._map_value(
                org_data.get('category') or pipedrive_data.get('category', 'Bens de Consumo'),
                cls.SEGMENT_MAPPING,
                'Bens de Consumo'
            ),
            'faturamento_anual_milhoes': cls._extract_revenue(
                org_data.get('annual_revenue') or pipedrive_data.get('value', 0)
            ),
            'numero_SKUs': org_data.get('sku_count', 100),  # Default estimate
            'margem_media_setor': cls._estimate_sector_margin(
                org_data.get('category') or pipedrive_data.get('category', 'Bens de Consumo')
            ),
            'exporta': int(org_data.get('exports', False) or pipedrive_data.get('exports', False)),
            
            # Contact information
            'contact_role': cls._map_value(
                pipedrive_data.get('job_title', 'Analista de Preços'),
                cls.CONTACT_ROLE_MAPPING,
                'Analista de Preços'
            ),
            'lead_source': cls._map_value(
                pipedrive_data.get('source', 'Inbound (Site)'),
                cls.LEAD_SOURCE_MAPPING,
                'Inbound (Site)'
            ),
            'crm_stage': cls._map_value(
                pipedrive_data.get('stage', 'Novo'),
                cls.CRM_STAGE_MAPPING,
                'Novo'
            ),
            
            # Engagement metrics
            'emails_enviados': engagement_metrics['emails_sent'],
            'emails_abertos': engagement_metrics['emails_opened'],
            'emails_respondidos': engagement_metrics['emails_responded'],
            'reunioes_realizadas': engagement_metrics['meetings_held'],
            'download_whitepaper': engagement_metrics['whitepaper_downloads'],
            'demo_solicitada': engagement_metrics['demo_requested'],
            'problemas_reportados_precificacao': engagement_metrics['pricing_problems'],
            'urgencia_projeto': engagement_metrics['project_urgency'],
            'days_since_first_touch': cls._calculate_days_since_creation(
                pipedrive_data.get('add_time') or pipedrive_data.get('created_at')
            )
        }
        
        return mapped_data
    
    @classmethod
    def _map_value(cls, value: str, mapping: Dict[str, str], default: str) -> str:
        """Map a value using provided mapping dictionary."""
        return mapping.get(value, default)
    
    @classmethod
    def _extract_revenue(cls, value: Any) -> float:
        """Extract revenue value and convert to millions."""
        if isinstance(value, (int, float)):
            return float(value) / 1_000_000  # Convert to millions
        elif isinstance(value, str):
            try:
                return float(value.replace(',', '').replace('$', '')) / 1_000_000
            except ValueError:
                return 50.0  # Default estimate
        return 50.0
    
    @classmethod
    def _estimate_sector_margin(cls, sector: str) -> float:
        """Estimate sector margin based on industry."""
        margin_estimates = {
            'Energia & Utilities': 0.15,
            'Químicos & Plásticos': 0.12,
            'Alimentos & Bebidas': 0.08,
            'Metalurgia': 0.10,
            'Máquinas & Equipamentos': 0.14,
            'Construção': 0.06,
            'Bens de Consumo': 0.09
        }
        return margin_estimates.get(sector, 0.10)
    
    @classmethod
    def _calculate_engagement_metrics(cls, activities: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate engagement metrics from activities."""
        metrics = {
            'emails_sent': 0,
            'emails_opened': 0,
            'emails_responded': 0,
            'meetings_held': 0,
            'whitepaper_downloads': 0,
            'demo_requested': 0,
            'pricing_problems': 0,
            'project_urgency': 0
        }
        
        for activity in activities:
            activity_type = activity.get('type', '').lower()
            subject = activity.get('subject', '').lower()
            
            if 'email' in activity_type:
                metrics['emails_sent'] += 1
                if activity.get('done', False):
                    metrics['emails_opened'] += 1
                if 'reply' in subject or 'response' in subject:
                    metrics['emails_responded'] += 1
            
            elif 'meeting' in activity_type or 'call' in activity_type:
                if activity.get('done', False):
                    metrics['meetings_held'] += 1
            
            elif 'whitepaper' in subject or 'download' in subject:
                metrics['whitepaper_downloads'] = 1
            
            elif 'demo' in subject:
                metrics['demo_requested'] = 1
            
            elif 'pricing' in subject or 'price' in subject:
                metrics['pricing_problems'] = 1
            
            elif 'urgent' in subject or 'asap' in subject:
                metrics['project_urgency'] = 1
        
        return metrics
    
    @classmethod
    def _calculate_days_since_creation(cls, created_at: Optional[str]) -> int:
        """Calculate days since creation."""
        if not created_at:
            return 30  # Default estimate
        
        try:
            if isinstance(created_at, str):
                # Handle different datetime formats
                for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d']:
                    try:
                        created_date = datetime.strptime(created_at.split('.')[0], fmt)
                        break
                    except ValueError:
                        continue
                else:
                    return 30  # Default if parsing fails
            else:
                created_date = created_at
            
            days_diff = (datetime.now() - created_date).days
            return max(0, days_diff)
            
        except Exception:
            return 30  # Default estimate
