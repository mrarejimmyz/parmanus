"""
Visual Element Annotation System
Adds numbered boxes/overlays to screenshots for Llama 3.2 Vision element identification
"""

import asyncio
import base64
import json
import re
from io import BytesIO
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from app.logger import logger
from app.schema import Function, ToolCall


class VisualElementAnnotator:
    """Handles visual annotation of browser screenshots with numbered element boxes"""

    def __init__(self, agent):
        self.agent = agent

    async def get_annotated_screenshot_and_analyze(self, goal: str) -> Dict:
        """Get screenshot with numbered element annotations and analyze with vision"""
        logger.info(f"🎨 Creating annotated screenshot for vision analysis: {goal}")

        try:
            # Step 1: Get page elements and screenshot
            elements_and_screenshot = await self._get_elements_and_screenshot()

            if not elements_and_screenshot["success"]:
                return elements_and_screenshot

            # Step 2: Annotate screenshot with numbered boxes
            annotated_result = await self._annotate_screenshot_with_elements(
                elements_and_screenshot["screenshot"],
                elements_and_screenshot["elements"],
            )

            if not annotated_result["success"]:
                return annotated_result

            # Step 3: Analyze annotated screenshot with vision
            vision_analysis = await self._analyze_annotated_screenshot(
                annotated_result["annotated_screenshot"],
                elements_and_screenshot["elements"],
                goal,
            )

            return {
                "success": True,
                "annotated_screenshot": annotated_result["annotated_screenshot"],
                "elements": elements_and_screenshot["elements"],
                "analysis": vision_analysis,
                "element_map": annotated_result["element_map"],
            }

        except Exception as e:
            logger.error(f"❌ Annotated screenshot analysis failed: {str(e)}")
            return {"success": False, "error": str(e)}

    async def find_element_by_description(self, element_description: str) -> Dict:
        """Find element using visual annotation and return its index"""
        logger.info(f"🔍 Finding element with visual annotation: {element_description}")

        goal = f"""
        Look at this annotated screenshot where clickable elements are marked with numbered boxes.

        Find the element that matches: {element_description}

        Please respond with:
        1. The NUMBER of the element (e.g., "Element 3" or just "3")
        2. A brief description of why this element matches
        3. Any other relevant observations

        Be specific about the element number.
        """

        result = await self.get_annotated_screenshot_and_analyze(goal)

        if result["success"]:
            # Extract element number from vision response
            element_number = self._extract_element_number(result["analysis"])

            if element_number is not None:
                # Map to actual browser element index
                browser_index = self._map_to_browser_index(
                    element_number, result["element_map"]
                )

                return {
                    "success": True,
                    "element_found": True,
                    "element_number": element_number,
                    "browser_index": browser_index,
                    "analysis": result["analysis"],
                    "annotated_screenshot": result["annotated_screenshot"],
                }
            else:
                return {
                    "success": True,
                    "element_found": False,
                    "analysis": result["analysis"],
                    "annotated_screenshot": result["annotated_screenshot"],
                }
        else:
            return result

    async def _get_elements_and_screenshot(self) -> Dict:
        """Get both interactive elements and screenshot from browser"""
        logger.info("📋 Getting page elements and screenshot")

        try:
            # Use extract_content which provides both content and screenshot
            tool_call = self._create_tool_call(
                "browser_use",
                {
                    "action": "extract_content",
                    "goal": "Get page elements and screenshot for visual annotation",
                },
            )

            result = await self.agent.execute_tool(tool_call)

            if hasattr(result, "error") and result.error:
                return {
                    "success": False,
                    "error": f"Browser extraction failed: {result.error}",
                }

            # Extract screenshot
            screenshot_data = self._extract_screenshot_data(result)
            if not screenshot_data:
                return {"success": False, "error": "No screenshot data found"}

            # Extract elements from page content
            elements = self._extract_elements_from_content(result)

            return {
                "success": True,
                "screenshot": screenshot_data,
                "elements": elements,
            }

        except Exception as e:
            logger.error(f"❌ Failed to get elements and screenshot: {str(e)}")
            return {"success": False, "error": str(e)}

    async def _annotate_screenshot_with_elements(
        self, screenshot_base64: str, elements: List[Dict]
    ) -> Dict:
        """Draw numbered boxes on screenshot for each interactive element"""
        logger.info(f"🎨 Annotating screenshot with {len(elements)} elements")

        try:
            # Decode base64 screenshot
            screenshot_bytes = base64.b64decode(screenshot_base64)
            image = Image.open(BytesIO(screenshot_bytes))

            # Create drawing context
            draw = ImageDraw.Draw(image)

            # Try to load a font, fallback to default if not available
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()

            element_map = {}

            # Draw numbered boxes for each element
            for i, element in enumerate(elements):
                element_number = i + 1  # Start numbering from 1
                element_map[element_number] = element.get("index", i)

                # Get element bounds (approximate if not available)
                bounds = element.get(
                    "bounds",
                    self._estimate_element_bounds(i, len(elements), image.size),
                )

                # Draw rectangle around element
                x1, y1 = bounds["x"], bounds["y"]
                x2, y2 = x1 + bounds["width"], y1 + bounds["height"]

                # Draw red rectangle
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

                # Draw number label with background
                label = str(element_number)
                label_bbox = draw.textbbox((0, 0), label, font=font)
                label_width = label_bbox[2] - label_bbox[0]
                label_height = label_bbox[3] - label_bbox[1]

                # Position label at top-left of element
                label_x = x1
                label_y = max(0, y1 - label_height - 5)

                # Draw label background
                draw.rectangle(
                    [
                        label_x,
                        label_y,
                        label_x + label_width + 6,
                        label_y + label_height + 4,
                    ],
                    fill="red",
                )

                # Draw label text
                draw.text((label_x + 3, label_y + 2), label, fill="white", font=font)

                logger.info(
                    f"📍 Annotated element {element_number}: {element.get('type', 'unknown')} at ({x1}, {y1})"
                )

            # Convert back to base64
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            annotated_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            return {
                "success": True,
                "annotated_screenshot": annotated_base64,
                "element_map": element_map,
            }

        except Exception as e:
            logger.error(f"❌ Screenshot annotation failed: {str(e)}")
            return {"success": False, "error": str(e)}

    async def _analyze_annotated_screenshot(
        self, screenshot_base64: str, elements: List[Dict], goal: str
    ) -> Dict:
        """Analyze annotated screenshot using Llama 3.2 Vision model"""
        try:
            # Create vision tool call
            vision_call = self._create_tool_call(
                "browser_use",
                {
                    "action": "vision_analyze",
                    "image": screenshot_base64,
                    "prompt": goal,
                    "model": "llama3.2-vision",  # Explicitly specify vision model
                    "options": {
                        "num_elements": len(elements),
                        "element_types": [
                            elem.get("type", "unknown") for elem in elements
                        ],
                        "require_element_number": True,  # Force model to identify specific element numbers
                    },
                },
            )

            # Execute vision analysis
            vision_result = await self.agent.execute_tool(vision_call)

            if hasattr(vision_result, "error") and vision_result.error:
                logger.error(f"❌ Vision analysis failed: {vision_result.error}")
                return {"success": False, "error": str(vision_result.error)}

            return {
                "success": True,
                "analysis": (
                    vision_result.output if hasattr(vision_result, "output") else None
                ),
                "elements": elements,
            }

        except Exception as e:
            logger.error(f"❌ Vision analysis failed: {str(e)}")
            return {"success": False, "error": str(e)}

    def _extract_elements_from_content(self, browser_result) -> List[Dict]:
        """Extract interactive elements from browser content"""
        try:
            # This is a simplified version - in reality, we'd need the browser tool
            # to provide structured element data with bounds

            # For now, create mock elements based on common page elements
            # In a real implementation, this would come from the browser tool
            mock_elements = [
                {
                    "index": 0,
                    "type": "input",
                    "text": "search",
                    "bounds": {"x": 300, "y": 200, "width": 400, "height": 40},
                },
                {
                    "index": 1,
                    "type": "button",
                    "text": "Google Search",
                    "bounds": {"x": 350, "y": 250, "width": 120, "height": 35},
                },
                {
                    "index": 2,
                    "type": "button",
                    "text": "I'm Feeling Lucky",
                    "bounds": {"x": 480, "y": 250, "width": 140, "height": 35},
                },
                {
                    "index": 3,
                    "type": "link",
                    "text": "Gmail",
                    "bounds": {"x": 600, "y": 50, "width": 50, "height": 20},
                },
                {
                    "index": 4,
                    "type": "link",
                    "text": "Images",
                    "bounds": {"x": 660, "y": 50, "width": 60, "height": 20},
                },
            ]

            logger.info(f"📋 Extracted {len(mock_elements)} interactive elements")
            return mock_elements

        except Exception as e:
            logger.error(f"❌ Element extraction failed: {str(e)}")
            return []

    def _estimate_element_bounds(
        self, index: int, total_elements: int, image_size: Tuple[int, int]
    ) -> Dict:
        """Estimate element bounds when not provided by browser"""
        width, height = image_size

        # Simple estimation - distribute elements across the page
        element_height = 40
        element_width = min(300, width // 3)

        # Estimate position based on index
        x = (width - element_width) // 2
        y = (height // 4) + (index * (element_height + 10))

        return {"x": x, "y": y, "width": element_width, "height": element_height}

    def _extract_element_number(self, vision_response: str) -> Optional[int]:
        """Extract element number from vision model response"""
        try:
            # Look for patterns like "Element 3", "element 3", "3", "#3", etc.
            patterns = [
                r"element\s+(\d+)",
                r"#(\d+)",
                r"number\s+(\d+)",
                r"box\s+(\d+)",
                r"\b(\d+)\b",
            ]

            for pattern in patterns:
                match = re.search(pattern, vision_response.lower())
                if match:
                    return int(match.group(1))

            return None

        except Exception as e:
            logger.error(f"❌ Failed to extract element number: {str(e)}")
            return None

    def _map_to_browser_index(
        self, element_number: int, element_map: Dict
    ) -> Optional[int]:
        """Map visual element number to browser element index"""
        return element_map.get(element_number)

    def _extract_screenshot_data(self, browser_result) -> Optional[str]:
        """Extract base64 screenshot data from browser result"""
        try:
            if hasattr(browser_result, "base64_image") and browser_result.base64_image:
                return browser_result.base64_image
            return None
        except Exception as e:
            logger.error(f"❌ Screenshot extraction failed: {str(e)}")
            return None

    def _create_tool_call(self, tool_name: str, args: Dict) -> ToolCall:
        """Create a tool call with proper formatting"""
        return ToolCall(
            id=f"vision_{int(asyncio.get_event_loop().time() * 1000)}",
            type="function",
            function=Function(name=tool_name, arguments=json.dumps(args)),
        )
