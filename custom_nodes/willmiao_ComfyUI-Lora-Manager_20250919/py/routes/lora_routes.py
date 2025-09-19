import asyncio
import logging
from aiohttp import web
from typing import Dict
from server import PromptServer  # type: ignore

from .base_model_routes import BaseModelRoutes
from ..services.lora_service import LoraService
from ..services.service_registry import ServiceRegistry
from ..services.metadata_service import get_default_metadata_provider
from ..utils.utils import get_lora_info

logger = logging.getLogger(__name__)

class LoraRoutes(BaseModelRoutes):
    """LoRA-specific route controller"""
    
    def __init__(self):
        """Initialize LoRA routes with LoRA service"""
        # Service will be initialized later via setup_routes
        self.service = None
        self.template_name = "loras.html"
    
    async def initialize_services(self):
        """Initialize services from ServiceRegistry"""
        lora_scanner = await ServiceRegistry.get_lora_scanner()
        self.service = LoraService(lora_scanner)
        
        # Initialize parent with the service
        super().__init__(self.service)
    
    def setup_routes(self, app: web.Application):
        """Setup LoRA routes"""
        # Schedule service initialization on app startup
        app.on_startup.append(lambda _: self.initialize_services())
        
        # Setup common routes with 'loras' prefix (includes page route)
        super().setup_routes(app, 'loras')
    
    def setup_specific_routes(self, app: web.Application, prefix: str):
        """Setup LoRA-specific routes"""
        # LoRA-specific query routes
        app.router.add_get(f'/api/lm/{prefix}/letter-counts', self.get_letter_counts)
        app.router.add_get(f'/api/lm/{prefix}/get-trigger-words', self.get_lora_trigger_words)
        app.router.add_get(f'/api/lm/{prefix}/usage-tips-by-path', self.get_lora_usage_tips_by_path)
        
        # ComfyUI integration
        app.router.add_post(f'/api/lm/{prefix}/get_trigger_words', self.get_trigger_words)
    
    def _parse_specific_params(self, request: web.Request) -> Dict:
        """Parse LoRA-specific parameters"""
        params = {}
        
        # LoRA-specific parameters
        if 'first_letter' in request.query:
            params['first_letter'] = request.query.get('first_letter')
        
        # Handle fuzzy search parameter name variation
        if request.query.get('fuzzy') == 'true':
            params['fuzzy_search'] = True
        
        # Handle additional filter parameters for LoRAs
        if 'lora_hash' in request.query:
            if not params.get('hash_filters'):
                params['hash_filters'] = {}
            params['hash_filters']['single_hash'] = request.query['lora_hash'].lower()
        elif 'lora_hashes' in request.query:
            if not params.get('hash_filters'):
                params['hash_filters'] = {}
            params['hash_filters']['multiple_hashes'] = [h.lower() for h in request.query['lora_hashes'].split(',')]
        
        return params
    
    def _validate_civitai_model_type(self, model_type: str) -> bool:
        """Validate CivitAI model type for LoRA"""
        from ..utils.constants import VALID_LORA_TYPES
        return model_type.lower() in VALID_LORA_TYPES
    
    def _get_expected_model_types(self) -> str:
        """Get expected model types string for error messages"""
        return "LORA, LoCon, or DORA"
    
    # LoRA-specific route handlers
    async def get_letter_counts(self, request: web.Request) -> web.Response:
        """Get count of LoRAs for each letter of the alphabet"""
        try:
            letter_counts = await self.service.get_letter_counts()
            return web.json_response({
                'success': True,
                'letter_counts': letter_counts
            })
        except Exception as e:
            logger.error(f"Error getting letter counts: {e}")
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
    
    async def get_lora_notes(self, request: web.Request) -> web.Response:
        """Get notes for a specific LoRA file"""
        try:
            lora_name = request.query.get('name')
            if not lora_name:
                return web.Response(text='Lora file name is required', status=400)
            
            notes = await self.service.get_lora_notes(lora_name)
            if notes is not None:
                return web.json_response({
                    'success': True,
                    'notes': notes
                })
            else:
                return web.json_response({
                    'success': False,
                    'error': 'LoRA not found in cache'
                }, status=404)
                
        except Exception as e:
            logger.error(f"Error getting lora notes: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
    
    async def get_lora_trigger_words(self, request: web.Request) -> web.Response:
        """Get trigger words for a specific LoRA file"""
        try:
            lora_name = request.query.get('name')
            if not lora_name:
                return web.Response(text='Lora file name is required', status=400)
            
            trigger_words = await self.service.get_lora_trigger_words(lora_name)
            return web.json_response({
                'success': True,
                'trigger_words': trigger_words
            })
            
        except Exception as e:
            logger.error(f"Error getting lora trigger words: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
    
    async def get_lora_usage_tips_by_path(self, request: web.Request) -> web.Response:
        """Get usage tips for a LoRA by its relative path"""
        try:
            relative_path = request.query.get('relative_path')
            if not relative_path:
                return web.Response(text='Relative path is required', status=400)
            
            usage_tips = await self.service.get_lora_usage_tips_by_relative_path(relative_path)
            return web.json_response({
                'success': True,
                'usage_tips': usage_tips or ''
            })
            
        except Exception as e:
            logger.error(f"Error getting lora usage tips by path: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
    
    async def get_lora_preview_url(self, request: web.Request) -> web.Response:
        """Get the static preview URL for a LoRA file"""
        try:
            lora_name = request.query.get('name')
            if not lora_name:
                return web.Response(text='Lora file name is required', status=400)
            
            preview_url = await self.service.get_lora_preview_url(lora_name)
            if preview_url:
                return web.json_response({
                    'success': True,
                    'preview_url': preview_url
                })
            else:
                return web.json_response({
                    'success': False,
                    'error': 'No preview URL found for the specified lora'
                }, status=404)
                
        except Exception as e:
            logger.error(f"Error getting lora preview URL: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
    
    async def get_lora_civitai_url(self, request: web.Request) -> web.Response:
        """Get the Civitai URL for a LoRA file"""
        try:
            lora_name = request.query.get('name')
            if not lora_name:
                return web.Response(text='Lora file name is required', status=400)
            
            result = await self.service.get_lora_civitai_url(lora_name)
            if result['civitai_url']:
                return web.json_response({
                    'success': True,
                    **result
                })
            else:
                return web.json_response({
                    'success': False,
                    'error': 'No Civitai data found for the specified lora'
                }, status=404)
                
        except Exception as e:
            logger.error(f"Error getting lora Civitai URL: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
    
    async def get_trigger_words(self, request: web.Request) -> web.Response:
        """Get trigger words for specified LoRA models"""
        try:
            json_data = await request.json()
            lora_names = json_data.get("lora_names", [])
            node_ids = json_data.get("node_ids", [])
            
            all_trigger_words = []
            for lora_name in lora_names:
                _, trigger_words = get_lora_info(lora_name)
                all_trigger_words.extend(trigger_words)
            
            # Format the trigger words
            trigger_words_text = ",, ".join(all_trigger_words) if all_trigger_words else ""
            
            # Send update to all connected trigger word toggle nodes
            for node_id in node_ids:
                PromptServer.instance.send_sync("trigger_word_update", {
                    "id": node_id,
                    "message": trigger_words_text
                })
            
            return web.json_response({"success": True})

        except Exception as e:
            logger.error(f"Error getting trigger words: {e}")
            return web.json_response({
                "success": False,
                "error": str(e)
            }, status=500)
