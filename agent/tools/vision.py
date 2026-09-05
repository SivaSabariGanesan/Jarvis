"""
Vision analysis and element interaction tools for JARVIS V4.
Enforces confidence gating, bounding-box center validation, ambiguity detection,
and protected UI action confirmation.
"""

import logging
from typing import Optional, Dict, Any, List

from agent.config import settings
from agent.ai.vision import vision_engine
from agent.tools.screenshot import capture_screen_image
from agent.tools.mouse import click, get_screen_dimensions
from agent.tools.security import (
    security_validator,
    RiskLevel,
    SecurityViolation,
    ConfirmationRequired,
    SecurityAuditLogger,
)

logger = logging.getLogger("jarvis.tools.vision")


async def analyze_screen(query: Optional[str] = None) -> str:
    """
    Capture the current desktop screen and use the local vision model to describe visible UI elements.
    """
    if not security_validator.is_enabled():
        raise SecurityViolation("Computer control is paused.")

    SecurityAuditLogger.log_request("analyze_screen", {"query": query or ""}, RiskLevel.LOW)

    img = capture_screen_image()
    analysis = await vision_engine.analyze_image(img, custom_prompt=query)

    if not analysis.get("success"):
        err = analysis.get("error", "Local vision model is not configured.")
        return f"Unable to analyze screen: {err}"

    desc = analysis.get("description", "")
    elements = analysis.get("elements", [])

    print(f"\033[90m[VISION] Detected {len(elements)} UI elements.\033[0m", flush=True)

    if elements:
        labels = [f"'{e['label']}' ({e['type']})" for e in elements[:6] if e.get("label")]
        summary = f"{desc} Visible controls include: {', '.join(labels)}."
    else:
        summary = desc or "Screen analyzed. No interactive controls identified."

    SecurityAuditLogger.log_execution("analyze_screen", True, summary)
    return summary


async def click_element(element_label: str) -> str:
    """
    Visually identify a UI control by name/label on screen and click its center.
    Validates confidence, detects ambiguous targets, and requires confirmation for protected UI.
    """
    if not security_validator.is_enabled():
        raise SecurityViolation("Computer control is paused.")

    if not element_label or not element_label.strip():
        raise SecurityViolation("Element label cannot be empty.")

    target = element_label.strip().lower()

    # Check protected UI before visual search
    if security_validator.is_protected_ui_element(target):
        SecurityAuditLogger.log_request("click_element", {"element_label": element_label}, RiskLevel.HIGH)
        # Will be caught by security layer if unconfirmed
    else:
        SecurityAuditLogger.log_request("click_element", {"element_label": element_label}, RiskLevel.LOW)

    img = capture_screen_image()
    w, h = img.width, img.height

    prompt = f"Locate the UI element with label or purpose '{element_label}' on screen. Return bounding box and confidence."
    analysis = await vision_engine.analyze_image(img, custom_prompt=prompt)

    if not analysis.get("success"):
        err = analysis.get("error", "Local vision model is not configured.")
        return f"I cannot locate '{element_label}': {err}"

    elements: List[Dict[str, Any]] = analysis.get("elements", [])

    # Filter matching elements
    matches = []
    for el in elements:
        label = el.get("label", "").lower()
        if target in label or label in target:
            matches.append(el)

    if not matches:
        return f"I couldn't find any visible UI element matching '{element_label}' on your screen."

    # Ambiguity Check
    high_conf_matches = [m for m in matches if m.get("confidence", 0.0) >= settings.vision_min_confidence]
    if len(high_conf_matches) > 1:
        return f"I found {len(high_conf_matches)} '{element_label}' elements on screen. Which one do you mean?"

    best_match = matches[0]
    confidence = float(best_match.get("confidence", 0.0))

    # Confidence Threshold Check
    if confidence < settings.vision_min_confidence:
        print(f"\033[91m[SECURITY] Vision action blocked: confidence {confidence:.2f} < {settings.vision_min_confidence:.2f}\033[0m", flush=True)
        return f"I can't confidently identify '{element_label}' on screen (confidence: {int(confidence*100)}%)."

    # Protected Action Confirmation Check
    if best_match.get("is_protected") or security_validator.is_protected_ui_element(best_match.get("label", "")):
        raise ConfirmationRequired(
            f"Planned action: Click '{best_match.get('label')}' on screen. This is a protected/destructive control. Do you want me to proceed, sir?",
            tool_name="click_element",
            arguments={"element_label": element_label},
        )

    # Coordinate Validation & Execution
    center = best_match.get("center", [0, 0])
    target_x, target_y = security_validator.validate_mouse_coordinates(center[0], center[1], w, h)

    click(target_x, target_y)
    summary = f"Clicked '{best_match.get('label')}' at ({target_x}, {target_y})."
    SecurityAuditLogger.log_execution("click_element", True, summary)
    return summary
