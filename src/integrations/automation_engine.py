"""
Automation engine for LeadScore AI Pipedrive integration.
Handles business logic and automated actions based on lead scores.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio
from enum import Enum

from .pipedrive_client import PipedriveClient

logger = logging.getLogger(__name__)


class Priority(Enum):
    """Lead priority levels."""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class AutomationEngine:
    """
    Handles automated actions based on lead scoring results.
    """
    
    def __init__(self, pipedrive_token: Optional[str], pipedrive_domain: str):
        """
        Initialize automation engine.
        
        Args:
            pipedrive_token: Pipedrive API token
            pipedrive_domain: Pipedrive company domain
        """
        self.pipedrive_client = None
        if pipedrive_token:
            self.pipedrive_client = PipedriveClient(pipedrive_token, pipedrive_domain)
        
        # Automation rules configuration
        self.automation_rules = self._load_automation_rules()
        
        # Custom field mappings (these would be configured in Pipedrive)
        self.custom_fields = {
            'lead_score': 'lead_score',
            'lead_priority': 'lead_priority',
            'conversion_probability': 'conversion_probability',
            'lead_quality_score': 'lead_quality_score',
            'last_scored_date': 'last_scored_date',
            'scoring_model_version': 'scoring_model_version'
        }
    
    def _load_automation_rules(self) -> Dict[str, Any]:
        """
        Load automation rules configuration.
        
        Returns:
            Automation rules dictionary
        """
        return {
            Priority.HIGH.value: {
                'assignment': {
                    'user_type': 'senior_sales_rep',
                    'priority_level': 'urgent'
                },
                'activities': [
                    {
                        'type': 'call',
                        'subject': 'High Priority Lead - Immediate Follow-up Required',
                        'due_time': 2,  # hours
                        'note': 'This lead has been scored as HIGH PRIORITY with {probability:.1%} conversion probability. Please contact immediately.'
                    },
                    {
                        'type': 'meeting',
                        'subject': 'Discovery Call - High Value Prospect',
                        'due_time': 24,  # hours
                        'note': 'Schedule discovery call with this high-value prospect.'
                    }
                ],
                'notifications': [
                    {
                        'type': 'email',
                        'recipients': ['sales_manager'],
                        'subject': 'High Priority Lead Alert',
                        'template': 'high_priority_alert'
                    }
                ],
                'pipeline_actions': {
                    'move_to_stage': 'qualified',
                    'add_labels': ['high-priority', 'hot-lead']
                }
            },
            Priority.MEDIUM.value: {
                'assignment': {
                    'user_type': 'sales_rep',
                    'priority_level': 'normal'
                },
                'activities': [
                    {
                        'type': 'call',
                        'subject': 'Medium Priority Lead - Follow-up',
                        'due_time': 24,  # hours
                        'note': 'This lead has been scored as MEDIUM PRIORITY with {probability:.1%} conversion probability. Please follow up within 24 hours.'
                    },
                    {
                        'type': 'email',
                        'subject': 'Qualification Email',
                        'due_time': 4,  # hours
                        'note': 'Send qualification email to assess needs and timeline.'
                    }
                ],
                'pipeline_actions': {
                    'add_labels': ['medium-priority', 'qualified-lead']
                }
            },
            Priority.LOW.value: {
                'assignment': {
                    'user_type': 'sdr',
                    'priority_level': 'low'
                },
                'activities': [
                    {
                        'type': 'email',
                        'subject': 'Low Priority Lead - Nurturing Sequence',
                        'due_time': 168,  # 1 week
                        'note': 'This lead has been scored as LOW PRIORITY with {probability:.1%} conversion probability. Add to nurturing sequence.'
                    }
                ],
                'pipeline_actions': {
                    'add_labels': ['low-priority', 'nurturing']
                }
            }
        }
    
    async def get_enriched_lead_data(self, object_id: int, object_type: str) -> Dict[str, Any]:
        """
        Get enriched lead data from Pipedrive.
        
        Args:
            object_id: Object ID in Pipedrive
            object_type: Type of object (person, deal, lead)
            
        Returns:
            Enriched lead data
        """
        if not self.pipedrive_client:
            logger.warning("Pipedrive client not configured, returning empty data")
            return {'activities': []}
        
        try:
            activities = []
            
            if object_type == 'person':
                activities = self.pipedrive_client.get_person_activities(object_id)
            elif object_type == 'deal':
                activities = self.pipedrive_client.get_deal_activities(object_id)
            # Note: Leads don't have activities in the same way
            
            return {
                'activities': activities,
                'enriched_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error enriching lead data: {str(e)}")
            return {'activities': []}
    
    async def execute_automation(self, object_id: int, object_type: str, 
                                scoring_result: Dict[str, Any], 
                                original_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute automation based on scoring results.
        
        Args:
            object_id: Object ID in Pipedrive
            object_type: Type of object (person, deal, lead)
            scoring_result: Lead scoring results
            original_data: Original webhook data
            
        Returns:
            Automation execution results
        """
        if not self.pipedrive_client:
            logger.warning("Pipedrive client not configured, skipping automation")
            return {'status': 'skipped', 'reason': 'no_pipedrive_client'}
        
        try:
            priority = scoring_result['priority']
            probability = scoring_result['conversion_probability']
            
            logger.info(f"Executing automation for {object_type} {object_id}: {priority} priority ({probability:.1%})")
            
            automation_results = {
                'object_id': object_id,
                'object_type': object_type,
                'priority': priority,
                'probability': probability,
                'actions_executed': [],
                'errors': []
            }
            
            # Update object with scoring data
            update_result = await self._update_object_with_scores(
                object_id, object_type, scoring_result
            )
            automation_results['actions_executed'].append(update_result)
            
            # Execute priority-based automation
            if priority in self.automation_rules:
                rules = self.automation_rules[priority]
                
                # Create activities
                if 'activities' in rules:
                    activity_results = await self._create_activities(
                        object_id, object_type, rules['activities'], scoring_result
                    )
                    automation_results['actions_executed'].extend(activity_results)
                
                # Send notifications
                if 'notifications' in rules:
                    notification_results = await self._send_notifications(
                        object_id, object_type, rules['notifications'], scoring_result, original_data
                    )
                    automation_results['actions_executed'].extend(notification_results)
                
                # Execute pipeline actions
                if 'pipeline_actions' in rules:
                    pipeline_results = await self._execute_pipeline_actions(
                        object_id, object_type, rules['pipeline_actions'], scoring_result
                    )
                    automation_results['actions_executed'].extend(pipeline_results)
            
            automation_results['status'] = 'completed'
            automation_results['executed_at'] = datetime.now().isoformat()
            
            return automation_results
            
        except Exception as e:
            logger.error(f"Error executing automation: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'object_id': object_id,
                'object_type': object_type
            }
    
    async def _update_object_with_scores(self, object_id: int, object_type: str, 
                                       scoring_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update Pipedrive object with scoring data.
        
        Args:
            object_id: Object ID
            object_type: Object type
            scoring_result: Scoring results
            
        Returns:
            Update result
        """
        try:
            update_data = {
                self.custom_fields['lead_score']: scoring_result['conversion_probability'],
                self.custom_fields['lead_priority']: scoring_result['priority'],
                self.custom_fields['conversion_probability']: f"{scoring_result['conversion_probability']:.1%}",
                self.custom_fields['lead_quality_score']: scoring_result.get('scores', {}).get('lead_quality_score', 0),
                self.custom_fields['last_scored_date']: datetime.now().strftime('%Y-%m-%d'),
                self.custom_fields['scoring_model_version']: scoring_result.get('model_info', {}).get('version', '1.0.0')
            }
            
            if object_type == 'person':
                result = self.pipedrive_client.update_person(object_id, update_data)
            elif object_type == 'deal':
                result = self.pipedrive_client.update_deal(object_id, update_data)
            elif object_type == 'lead':
                result = self.pipedrive_client.update_lead(str(object_id), update_data)
            else:
                raise ValueError(f"Unsupported object type: {object_type}")
            
            return {
                'action': 'update_scores',
                'status': 'success',
                'object_id': object_id,
                'object_type': object_type,
                'updated_fields': list(update_data.keys())
            }
            
        except Exception as e:
            logger.error(f"Error updating object with scores: {str(e)}")
            return {
                'action': 'update_scores',
                'status': 'failed',
                'error': str(e),
                'object_id': object_id,
                'object_type': object_type
            }
    
    async def _create_activities(self, object_id: int, object_type: str, 
                               activities_config: List[Dict[str, Any]], 
                               scoring_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create activities based on configuration.
        
        Args:
            object_id: Object ID
            object_type: Object type
            activities_config: Activities configuration
            scoring_result: Scoring results
            
        Returns:
            List of activity creation results
        """
        results = []
        
        for activity_config in activities_config:
            try:
                # Calculate due date
                due_hours = activity_config.get('due_time', 24)
                due_date = datetime.now() + timedelta(hours=due_hours)
                
                # Format note with scoring data
                note = activity_config.get('note', '').format(
                    probability=scoring_result['conversion_probability'],
                    priority=scoring_result['priority'],
                    confidence=scoring_result.get('confidence', 0)
                )
                
                activity_data = {
                    'subject': activity_config['subject'],
                    'type': activity_config['type'],
                    'due_date': due_date.strftime('%Y-%m-%d'),
                    'due_time': due_date.strftime('%H:%M'),
                    'note': note
                }
                
                # Link to appropriate object
                if object_type == 'person':
                    activity_data['person_id'] = object_id
                elif object_type == 'deal':
                    activity_data['deal_id'] = object_id
                
                result = self.pipedrive_client.create_activity(activity_data)
                
                results.append({
                    'action': 'create_activity',
                    'status': 'success',
                    'activity_type': activity_config['type'],
                    'activity_id': result['data']['id'],
                    'due_date': due_date.isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error creating activity: {str(e)}")
                results.append({
                    'action': 'create_activity',
                    'status': 'failed',
                    'error': str(e),
                    'activity_type': activity_config.get('type', 'unknown')
                })
        
        return results
    
    async def _send_notifications(self, object_id: int, object_type: str,
                                notifications_config: List[Dict[str, Any]],
                                scoring_result: Dict[str, Any],
                                original_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Send notifications based on configuration.
        
        Args:
            object_id: Object ID
            object_type: Object type
            notifications_config: Notifications configuration
            scoring_result: Scoring results
            original_data: Original webhook data
            
        Returns:
            List of notification results
        """
        results = []
        
        for notification_config in notifications_config:
            try:
                # Create notification note in Pipedrive
                note_content = f"""
🚨 HIGH PRIORITY LEAD ALERT 🚨

Lead Score: {scoring_result['conversion_probability']:.1%}
Priority: {scoring_result['priority']}
Confidence: {scoring_result.get('confidence', 0):.1%}

Object: {object_type.title()} #{object_id}
Company: {original_data.get('org_name', 'Unknown')}
Contact: {original_data.get('name', 'Unknown')}

Action Required: Immediate follow-up recommended for this high-value prospect.

Generated by LeadScore AI at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                """.strip()
                
                note_data = {
                    'content': note_content,
                    'pinned_to_deal_flag': 1 if object_type == 'deal' else 0,
                    'pinned_to_person_flag': 1 if object_type == 'person' else 0
                }
                
                if object_type == 'person':
                    note_data['person_id'] = object_id
                elif object_type == 'deal':
                    note_data['deal_id'] = object_id
                
                result = self.pipedrive_client.create_note(note_data)
                
                results.append({
                    'action': 'send_notification',
                    'status': 'success',
                    'notification_type': notification_config['type'],
                    'note_id': result['data']['id']
                })
                
            except Exception as e:
                logger.error(f"Error sending notification: {str(e)}")
                results.append({
                    'action': 'send_notification',
                    'status': 'failed',
                    'error': str(e),
                    'notification_type': notification_config.get('type', 'unknown')
                })
        
        return results
    
    async def _execute_pipeline_actions(self, object_id: int, object_type: str,
                                      pipeline_config: Dict[str, Any],
                                      scoring_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute pipeline actions based on configuration.
        
        Args:
            object_id: Object ID
            object_type: Object type
            pipeline_config: Pipeline configuration
            scoring_result: Scoring results
            
        Returns:
            List of pipeline action results
        """
        results = []
        
        # Note: This is a simplified implementation
        # In a real scenario, you'd need to map stage names to IDs
        # and handle label/tag management through custom fields or notes
        
        try:
            update_data = {}
            
            # Add labels as a note or custom field
            if 'add_labels' in pipeline_config:
                labels = pipeline_config['add_labels']
                label_note = f"Labels: {', '.join(labels)} (Added by LeadScore AI)"
                
                note_data = {
                    'content': label_note
                }
                
                if object_type == 'person':
                    note_data['person_id'] = object_id
                elif object_type == 'deal':
                    note_data['deal_id'] = object_id
                
                self.pipedrive_client.create_note(note_data)
                
                results.append({
                    'action': 'add_labels',
                    'status': 'success',
                    'labels': labels
                })
            
            # Stage movement would require stage ID mapping
            if 'move_to_stage' in pipeline_config:
                # This would need proper stage ID resolution
                results.append({
                    'action': 'move_to_stage',
                    'status': 'skipped',
                    'reason': 'stage_id_mapping_required',
                    'target_stage': pipeline_config['move_to_stage']
                })
            
        except Exception as e:
            logger.error(f"Error executing pipeline actions: {str(e)}")
            results.append({
                'action': 'pipeline_actions',
                'status': 'failed',
                'error': str(e)
            })
        
        return results
    
    def get_automation_summary(self) -> Dict[str, Any]:
        """
        Get summary of automation configuration.
        
        Returns:
            Automation summary
        """
        return {
            'pipedrive_connected': self.pipedrive_client is not None,
            'supported_priorities': list(self.automation_rules.keys()),
            'custom_fields': self.custom_fields,
            'automation_features': {
                'score_updates': True,
                'activity_creation': True,
                'notifications': True,
                'pipeline_actions': True
            }
        }
