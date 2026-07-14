"""
AutoQA — AI Browser Testing Agent
LangGraph pipeline:
  Parse Test → Plan Steps → Execute Steps (Playwright) → Vision Verify (Gemini) → Report
"""

import os
import json
from typing import TypedDict, Annotated, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.tools.browser import BrowserSession, is_url_safe
from src.tools.vision import verify_screenshot
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─── State ────────────────────────────────────────────────────────────────────

class TestState(TypedDict):
    messages: Annotated[list, add_messages]
    # Input
    test_description: str
    target_url: str
    # Parsed
    test_plan: Optional[dict]
    # Execution
    steps_executed: List[dict]
    screenshots: List[dict]
    vision_results: List[dict]
    # Results
    passed: Optional[bool]
    fail_reason: Optional[str]
    report: Optional[str]
    # Meta
    errors: List[str]
    status: str


# ─── LLM ──────────────────────────────────────────────────────────────────────

def get_llm(temperature=0.05):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        max_tokens=4096,
        api_key=api_key,
    )


def call_llm_json(system: str, user: str) -> dict:
    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=system + "\n\nRESPOND ONLY WITH VALID JSON. No markdown, no backticks."),
        HumanMessage(content=user),
    ])
    text = response.content.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


# ─── Node 1: Parse & Plan ─────────────────────────────────────────────────────

def plan_test_node(state: TestState) -> TestState:
    logger.info("🧠 Planning test steps from natural language...")

    # Safety check on URL
    if not is_url_safe(state["target_url"]):
        state["status"] = "blocked"
        state["errors"].append("URL blocked by safety policy: only public HTTP/HTTPS URLs allowed")
        state["passed"] = False
        state["fail_reason"] = "URL blocked by safety policy"
        return state

    try:
        plan = call_llm_json(
            system="""You are a senior QA engineer converting plain-English test descriptions into structured browser test steps.
You understand Playwright selectors and browser automation.
Be precise with selectors — prefer text-based ones when possible.
Safety: never generate steps that fill passwords into untrusted sites or perform destructive actions.""",
            user=f"""Convert this plain-English test into structured steps for browser automation:

URL: {state['target_url']}
TEST: {state['test_description']}

Return JSON:
{{
  "test_name": "short descriptive name",
  "objective": "what this test verifies",
  "steps": [
    {{
      "step_number": 1,
      "action": "navigate|click|click_text|type|press_key|wait_for_text|wait_for_selector|assert_text_visible|assert_url_contains|assert_title_contains|scroll_down",
      "target": "URL, selector, text, or key name",
      "value": "text to type (for type action) or null",
      "description": "what this step does in plain English",
      "take_vision_screenshot": true/false,
      "vision_check": "what Gemini should verify in this screenshot (or null)"
    }}
  ],
  "success_criteria": "how to determine if the test passed overall"
}}

IMPORTANT:
- First step should always be navigate to the URL
- Use text-based selectors like 'text=Submit' when possible
- Prefer assert_ steps to verify outcomes
- Mark important steps with take_vision_screenshot=true
- Keep to max 15 steps""",
        )

        state["test_plan"] = plan
        state["status"] = "planned"
        n = len(plan.get("steps", []))
        state["messages"].append(AIMessage(
            content=f"🧠 Test planned: '{plan.get('test_name')}' — {n} steps\n📋 Objective: {plan.get('objective')}"
        ))
        logger.info(f"Test planned: {n} steps")

    except Exception as e:
        state["errors"].append(f"Planning error: {e}")
        state["status"] = "error"
        state["passed"] = False
        state["fail_reason"] = f"Could not parse test: {e}"

    return state


# ─── Node 2: Execute Steps ────────────────────────────────────────────────────

def execute_test_node(state: TestState) -> TestState:
    logger.info("▶️ Executing test steps in browser...")
    plan = state["test_plan"]
    steps = plan.get("steps", [])
    session = BrowserSession(headless=True)

    try:
        session.start()
        results = []

        for step in steps:
            action = step["action"]
            target = step.get("target", "")
            value = step.get("value")

            # Execute the action
            if action == "navigate":
                result = session.navigate(target)
            elif action == "click":
                result = session.click(target)
            elif action == "click_text":
                result = session.click_text(target)
            elif action == "type":
                result = session.type_text(target, value or "")
            elif action == "press_key":
                result = session.press_key(target)
            elif action == "wait_for_text":
                result = session.wait_for_text(target)
            elif action == "wait_for_selector":
                result = session.wait_for_selector(target)
            elif action == "assert_text_visible":
                result = session.assert_text_visible(target)
            elif action == "assert_url_contains":
                result = session.assert_url_contains(target)
            elif action == "assert_title_contains":
                result = session.assert_title_contains(target)
            elif action == "scroll_down":
                result = session.scroll_down(int(target) if str(target).isdigit() else 500)
            else:
                result = {"action": action, "success": False, "error": f"Unknown action: {action}"}

            result["step_description"] = step.get("description", "")
            result["step_number"] = step.get("step_number", len(results) + 1)
            results.append(result)

            # Vision verification if requested
            if step.get("take_vision_screenshot") and session.screenshots:
                latest_screenshot = session.screenshots[-1]
                vision_check = step.get("vision_check")
                if vision_check and latest_screenshot.get("path"):
                    vision_result = verify_screenshot(
                        latest_screenshot["path"],
                        expected_description=vision_check,
                        question=f"Step {step['step_number']}: {step.get('description', '')}",
                    )
                    vision_result["step_number"] = step.get("step_number")
                    vision_result["step_description"] = step.get("description", "")
                    state["vision_results"].append(vision_result)

            # Stop on assertion failure
            if not result.get("success") and action.startswith("assert_"):
                state["fail_reason"] = f"Step {result['step_number']} failed: {result.get('error', 'Assertion failed')}"
                break

        state["steps_executed"] = results
        state["screenshots"] = session.screenshots

        # Determine pass/fail
        failed_steps = [r for r in results if not r.get("success")]
        assertion_fails = [r for r in failed_steps if "assert" in r.get("action", "")]

        if assertion_fails:
            state["passed"] = False
            state["fail_reason"] = state.get("fail_reason") or f"{len(assertion_fails)} assertion(s) failed"
        elif len(failed_steps) > len(results) * 0.5:
            state["passed"] = False
            state["fail_reason"] = f"{len(failed_steps)}/{len(results)} steps failed"
        else:
            state["passed"] = True
            state["fail_reason"] = None

        state["status"] = "executed"
        status_msg = "✅ PASSED" if state["passed"] else f"❌ FAILED: {state['fail_reason']}"
        state["messages"].append(AIMessage(
            content=f"{status_msg}\n{len(results) - len(failed_steps)}/{len(results)} steps succeeded"
        ))

    except Exception as e:
        state["errors"].append(f"Execution error: {e}")
        state["passed"] = False
        state["fail_reason"] = f"Browser error: {e}"
        state["status"] = "error"
    finally:
        session.stop()

    return state


# ─── Node 3: Generate Report ──────────────────────────────────────────────────

def report_node(state: TestState) -> TestState:
    logger.info("📝 Generating test report...")
    plan = state.get("test_plan", {})
    steps = state.get("steps_executed", [])
    vision = state.get("vision_results", [])
    passed = state.get("passed", False)

    passed_steps = sum(1 for s in steps if s.get("success"))
    failed_steps = [s for s in steps if not s.get("success")]

    # Step results table
    step_lines = []
    for s in steps:
        icon = "✅" if s.get("success") else "❌"
        step_lines.append(
            f"{icon} Step {s.get('step_number', '?')}: {s.get('step_description', s.get('action', ''))}"
            + (f"\n   ↳ Error: {s.get('error', '')}" if not s.get("success") else "")
        )

    # Vision results
    vision_lines = []
    for v in vision:
        icon = "👁️✅" if v.get("verified") else "👁️❌" if v.get("verified") is False else "👁️⚠️"
        vision_lines.append(
            f"{icon} Step {v.get('step_number')}: {v.get('description', '')}\n"
            f"   Confidence: {v.get('confidence', 'N/A')} | {v.get('matches', '')}"
        )

    verdict = "✅ PASSED" if passed else "❌ FAILED"
    report = f"""# 🤖 AutoQA Test Report

## {verdict}

**Test:** {plan.get('test_name', 'Browser Test')}
**URL:** {state['target_url']}
**Objective:** {plan.get('objective', '')}

---

## 📊 Results Summary

| Metric | Value |
|--------|-------|
| Status | {verdict} |
| Steps Passed | {passed_steps}/{len(steps)} |
| Vision Checks | {len(vision)} |
| Fail Reason | {state.get('fail_reason') or 'N/A'} |

---

## 🔢 Step-by-Step Results

{chr(10).join(step_lines)}

{f'''---

## 👁️ Gemini Vision Checks

{chr(10).join(vision_lines)}''' if vision_lines else ''}

---

## 🎯 Success Criteria

{plan.get('success_criteria', 'N/A')}

---
*Generated by AutoQA — LangGraph + Playwright + Gemini Vision*
"""

    state["report"] = report
    state["status"] = "complete"
    state["messages"].append(AIMessage(content=f"📝 Report generated — {verdict}"))
    return state


# ─── Build Graph ──────────────────────────────────────────────────────────────

def build_autoqa_graph():
    graph = StateGraph(TestState)
    graph.add_node("plan", plan_test_node)
    graph.add_node("execute", execute_test_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "report")
    graph.add_edge("report", END)

    return graph.compile()


def run_test(test_description: str, target_url: str) -> TestState:
    graph = build_autoqa_graph()
    initial: TestState = {
        "messages": [HumanMessage(content=test_description)],
        "test_description": test_description,
        "target_url": target_url,
        "test_plan": None,
        "steps_executed": [],
        "screenshots": [],
        "vision_results": [],
        "passed": None,
        "fail_reason": None,
        "report": None,
        "errors": [],
        "status": "start",
    }
    return graph.invoke(initial)
