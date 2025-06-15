"""
Vision-Enhanced Browser Interaction
Integrates Llama 3.2 Vision with browser automation for true visual interaction
"""

import asyncio
import base64
import json
from typing import Dict, List, Optional, Tuple
from app.logger import logger
from app.schema import ToolCall, Function, Message


class VisionBrowserInteraction:
    """Handles vision-enhanced browser interactions using Llama 3.2 Vision"""
    
    def __init__(self, agent):
        self.agent = agent
        
    async def take_screenshot_and_analyze(self, goal: str) -> Dict:
        """Take a screenshot and analyze it with Llama 3.2 Vision"""
        logger.info(f"📸 Taking screenshot for vision analysis: {goal}")
        
        try:
            # Get current browser state which includes screenshot
            screenshot_call = self._create_tool_call("browser_use", {
                "action": "get_current_state"
            })
            
            screenshot_result = await self.agent.execute_tool(screenshot_call)
            
            if hasattr(screenshot_result, 'error') and screenshot_result.error:
                return {"success": False, "error": f"Screenshot failed: {screenshot_result.error}"}
            
            # Extract base64 image from result
            screenshot_data = self._extract_screenshot_data(screenshot_result)
            
            if not screenshot_data:
                return {"success": False, "error": "No screenshot data found"}
            
            # Analyze screenshot with Llama 3.2 Vision
            vision_analysis = await self._analyze_with_vision(screenshot_data, goal)
            
            return {
                "success": True,
                "screenshot": screenshot_data,
                "analysis": vision_analysis
            }
            
        except Exception as e:
            logger.error(f"❌ Screenshot and analysis failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def find_element_visually(self, element_description: str) -> Dict:
        """Find an element on the page using vision analysis"""
        logger.info(f"👁️ Looking for element visually: {element_description}")
        
        goal = f"""
        Analyze this webpage screenshot and find the {element_description}.
        
        Please provide:
        1. Whether the element exists on the page
        2. A description of where it's located
        3. If possible, approximate coordinates or position description
        4. Any relevant details about the element's appearance
        
        Be specific and accurate in your analysis.
        """
        
        result = await self.take_screenshot_and_analyze(goal)
        
        if result["success"]:
            return {
                "success": True,
                "element_found": "search" in result["analysis"].lower() or "input" in result["analysis"].lower(),
                "location_description": result["analysis"],
                "screenshot": result["screenshot"]
            }
        else:
            return result
    
    async def verify_page_visually(self, expected_content: str) -> Dict:
        """Verify page content using vision analysis"""
        logger.info(f"🔍 Verifying page content visually: {expected_content}")
        
        goal = f"""
        Analyze this webpage screenshot and check if it contains: {expected_content}
        
        Please confirm:
        1. Is this the expected type of page?
        2. What key elements do you see?
        3. Does the page appear to have loaded correctly?
        4. Any issues or unexpected content?
        
        Provide a clear yes/no answer about whether this matches the expected content.
        """
        
        result = await self.take_screenshot_and_analyze(goal)
        
        if result["success"]:
            analysis = result["analysis"].lower()
            is_correct_page = any(keyword in analysis for keyword in expected_content.lower().split())
            
            return {
                "success": True,
                "is_correct_page": is_correct_page,
                "visual_analysis": result["analysis"],
                "screenshot": result["screenshot"]
            }
        else:
            return result
    
    async def get_interaction_guidance(self, action_goal: str) -> Dict:
        """Get guidance on how to interact with the page using vision"""
        logger.info(f"🎯 Getting interaction guidance: {action_goal}")
        
        goal = f"""
        Analyze this webpage screenshot to help with this action: {action_goal}
        
        Please provide specific guidance:
        1. What elements are available for interaction?
        2. Where should I click or type to achieve the goal?
        3. What's the best approach for this action?
        4. Are there any potential issues or obstacles?
        
        Be specific about locations and provide actionable advice.
        """
        
        result = await self.take_screenshot_and_analyze(goal)
        
        if result["success"]:
            return {
                "success": True,
                "guidance": result["analysis"],
                "screenshot": result["screenshot"]
            }
        else:
            return result
    
    async def _analyze_with_vision(self, screenshot_base64: str, goal: str) -> str:
        """Analyze screenshot using Llama 3.2 Vision"""
        logger.info("🧠 Analyzing screenshot with Llama 3.2 Vision")
        
        try:
            # Check if vision is available
            if not hasattr(self.agent, 'llm') or not hasattr(self.agent.llm, '_impl'):
                return "Vision analysis not available - LLM not properly initialized"
            
            llm_impl = self.agent.llm._impl
            
            if not hasattr(llm_impl, 'ask_vision') or not llm_impl.vision_enabled:
                return "Vision analysis not available - Vision not enabled in LLM configuration"
            
            # Prepare image data
            image_url = f"data:image/jpeg;base64,{screenshot_base64}"
            
            # Create vision prompt
            messages = [
                Message(
                    role="user",
                    content=goal
                )
            ]
            
            # Call vision model
            vision_response = await llm_impl.ask_vision(
                messages=messages,
                images=[image_url],
                temp=0.1  # Low temperature for consistent analysis
            )
            
            if vision_response and hasattr(vision_response, 'content'):
                return vision_response.content
            else:
                return "Vision analysis failed - No response from vision model"
                
        except Exception as e:
            logger.error(f"❌ Vision analysis error: {str(e)}")
            return f"Vision analysis failed: {str(e)}"
    
    def _extract_screenshot_data(self, screenshot_result) -> Optional[str]:
        """Extract base64 screenshot data from browser tool result"""
        try:
            logger.info(f"🔍 Extracting screenshot from result: {type(screenshot_result)}")
            
            # Handle different possible result formats
            if hasattr(screenshot_result, 'base64_image') and screenshot_result.base64_image:
                logger.info("✅ Found screenshot in base64_image attribute")
                return screenshot_result.base64_image
            
            if hasattr(screenshot_result, 'output'):
                output = screenshot_result.output
                logger.info(f"📊 Checking output field: {type(output)}")
                
                # If output is a string, try to parse as JSON
                if isinstance(output, str):
                    try:
                        parsed = json.loads(output)
                        if 'base64_image' in parsed:
                            logger.info("✅ Found screenshot in parsed output.base64_image")
                            return parsed['base64_image']
                        elif 'screenshot' in parsed:
                            logger.info("✅ Found screenshot in parsed output.screenshot")
                            return parsed['screenshot']
                    except json.JSONDecodeError:
                        # If not JSON, check if it's base64 data directly
                        if len(output) > 100 and output.replace('+', '').replace('/', '').replace('=', '').isalnum():
                            logger.info("✅ Found screenshot as direct base64 string")
                            return output
                
                # If output is a dict
                elif isinstance(output, dict):
                    if 'base64_image' in output:
                        logger.info("✅ Found screenshot in output dict.base64_image")
                        return output['base64_image']
                    elif 'screenshot' in output:
                        logger.info("✅ Found screenshot in output dict.screenshot")
                        return output['screenshot']
            
            # Check if result itself has screenshot data
            if hasattr(screenshot_result, 'screenshot'):
                logger.info("✅ Found screenshot in screenshot attribute")
                return screenshot_result.screenshot
            
            # Log the actual structure for debugging
            logger.warning(f"⚠️ No screenshot data found. Result structure: {dir(screenshot_result)}")
            if hasattr(screenshot_result, 'output'):
                logger.warning(f"⚠️ Output content preview: {str(screenshot_result.output)[:200]}...")
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error extracting screenshot data: {str(e)}")
            return None
    
    def _create_tool_call(self, function_name: str, arguments: Dict) -> ToolCall:
        """Create a properly formatted ToolCall object"""
        return ToolCall(
            id=f"call_{function_name}_{id(arguments)}",
            type="function",
            function=Function(name=function_name, arguments=json.dumps(arguments))
        )

