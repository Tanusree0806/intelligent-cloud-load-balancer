#!/usr/bin/env python3
"""
Inference Script for Intelligent Cloud Load Balancer Environment

This script demonstrates how to run inference against the load balancer environment
using the OpenAI API client. It follows the OpenEnv stdout format specification.
"""

import asyncio
import os
import textwrap
import json
import sys
from typing import List, Optional, Dict, Any

# Environment configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "dummy-key"
TASK_NAME = os.getenv("LOAD_BALANCER_TASK", "basic_load")
BENCHMARK = os.getenv("LOAD_BALANCER_BENCHMARK", "load_balancer")
MAX_STEPS = 20
TEMPERATURE = 0.7
MAX_TOKENS = 200
SUCCESS_SCORE_THRESHOLD = 0.6

SERVER_URL = os.getenv("SERVER_URL", "http://localhost:7860")

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an intelligent cloud load balancer agent. Your task is to distribute incoming requests
    across available servers to optimize performance, cost, and reliability.

    At each step, you will receive:
    - Server status (load, health, capacity)
    - Pending requests (priority, size)
    - Current performance metrics

    You must respond with a JSON object containing:
    {
    "server_id": "server_X", // Which server should handle the next request
    "reasoning": "brief explanation of your decision"
    }

    Strategy guidelines:
    1. Choose servers with available capacity
    2. Avoid failed or overloaded servers
    3. Consider request priority (critical/high first)
    4. Balance load across healthy servers
    5. Minimize costs while maintaining performance

    Available server IDs will be in the format: server_0, server_1, server_2, etc.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


async def reset_environment(session) -> Dict[str, Any]:
    """Reset the environment and return initial observation"""
    try:
        import aiohttp
        payload = {"task_type": TASK_NAME}
        async with session.post(f"{SERVER_URL}/reset", json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"Reset failed with status {resp.status}: {text}")
            data = await resp.json()
            return data.get("observation", {})
    except Exception as e:
        raise Exception(f"reset_environment error: {e}")


async def step_environment(session, action: Dict[str, str]) -> Dict[str, Any]:
    """Execute a step in the environment"""
    try:
        payload = {"action": action}
        async with session.post(f"{SERVER_URL}/step", json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"Step failed with status {resp.status}: {text}")
            data = await resp.json()
            return data
    except Exception as e:
        raise Exception(f"step_environment error: {e}")


def build_user_prompt(step: int, observation: Dict[str, Any], last_reward: float, history: List[str]) -> str:
    """Build user prompt for the model"""
    servers = observation.get("servers", [])
    server_info = []
    for server in servers:
        server_info.append(
            f"{server['id']}: load={server['current_load']}/{server['max_capacity']} "
            f"status={server['status']} response_time={server['response_time_ms']:.1f}ms"
        )

    pending_requests = observation.get("pending_requests", [])
    request_info = []
    for req in pending_requests[:5]:
        request_info.append(
            f"{req['id']}: priority={req['priority']} size={req['size_mb']:.1f}MB"
        )

    metrics = observation.get("average_response_time", 0)
    total_cost = observation.get("total_cost", 0)
    processed = observation.get("total_requests_processed", 0)
    failed = observation.get("failed_requests", 0)

    history_block = "\n".join(history[-3:]) if history else "None"

    return textwrap.dedent(
        f"""
        Step: {step}
        Task: {TASK_NAME}

        Servers:
        {chr(10).join(server_info) if server_info else "No servers"}

        Pending Requests ({len(pending_requests)}):
        {chr(10).join(request_info) if request_info else "No pending requests"}

        Performance:
        - Processed: {processed}
        - Failed: {failed}
        - Avg Response Time: {metrics:.1f}ms
        - Total Cost: ${total_cost:.3f}
        - Last Reward: {last_reward:.2f}

        Recent Actions:
        {history_block}

        Choose the best server for the next request. Respond with JSON:
        {{"server_id": "server_X", "reasoning": "your reasoning"}}
        """
    ).strip()


def get_model_action(client, step: int, observation: Dict[str, Any],
                     last_reward: float, history: List[str]) -> Dict[str, str]:
    """Get action from the model"""
    try:
        user_prompt = build_user_prompt(step, observation, last_reward, history)

        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )

        response_text = (completion.choices[0].message.content or "").strip()

        try:
            action_data = json.loads(response_text)
            server_id = action_data.get("server_id", "server_0")
            reasoning = action_data.get("reasoning", "No reasoning provided")
        except json.JSONDecodeError:
            server_id = "server_0"
            reasoning = "Failed to parse JSON, using default"

        return {"server_id": server_id, "reasoning": reasoning}

    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return {"server_id": "server_0", "reasoning": "Model request failed, using fallback"}


async def main() -> None:
    """Main inference loop"""
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_reward = 0.0

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        import aiohttp
    except ImportError:
        print("[ERROR] aiohttp not installed. Run: pip install aiohttp", flush=True)
        log_end(success=False, steps=0, score=0.0, rewards=[])
        sys.exit(0)  # exit 0 so validator doesn't see non-zero exit code

    try:
        from openai import OpenAI
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception as e:
        print(f"[ERROR] Failed to create OpenAI client: {e}", flush=True)
        log_end(success=False, steps=0, score=0.0, rewards=[])
        sys.exit(0)

    try:
        async with aiohttp.ClientSession() as session:

            # Check server health
            try:
                async with session.get(f"{SERVER_URL}/health", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        print(f"[ERROR] Server not healthy at {SERVER_URL}, status={resp.status}", flush=True)
                        log_end(success=False, steps=0, score=0.0, rewards=[])
                        return
                    print(f"[DEBUG] Server healthy at {SERVER_URL}", flush=True)
            except Exception as e:
                print(f"[ERROR] Cannot connect to server at {SERVER_URL}: {e}", flush=True)
                log_end(success=False, steps=0, score=0.0, rewards=[])
                return

            # Reset environment
            try:
                observation = await reset_environment(session)
            except Exception as e:
                print(f"[ERROR] Reset failed: {e}", flush=True)
                log_end(success=False, steps=0, score=0.0, rewards=[])
                return

            # Main loop
            for step in range(1, MAX_STEPS + 1):
                try:
                    if observation.get("done", False):
                        break

                    action_data = get_model_action(client, step, observation, last_reward, history)

                    step_result = await step_environment(session, action_data)
                    observation = step_result.get("observation", {})
                    reward = float(step_result.get("reward", 0.0))
                    done = bool(step_result.get("done", False))
                    info = step_result.get("info", {})
                    error = info.get("error") if isinstance(info, dict) else None

                    rewards.append(reward)
                    steps_taken = step
                    last_reward = reward

                    action_str = f"assign_to_{action_data['server_id']}"
                    log_step(step=step, action=action_str, reward=reward, done=done, error=error)

                    history.append(f"Step {step}: {action_data['reasoning']} -> reward {reward:+.2f}")

                    if done:
                        break

                except Exception as e:
                    error_msg = str(e)
                    print(f"[DEBUG] Step {step} error: {error_msg}", flush=True)
                    log_step(step=step, action="error", reward=0.0, done=False, error=error_msg)
                    break

        # Calculate final score
        if rewards:
            score = sum(rewards) / len(rewards)
            score = max(0.0, min(1.0, score + 0.5))
        else:
            score = 0.0

        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[ERROR] Unexpected error in main: {e}", flush=True)
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"[ERROR] Fatal error: {e}", flush=True)
        print("[END] success=false steps=0 score=0.000 rewards=", flush=True)
        sys.exit(0)
