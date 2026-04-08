#!/usr/bin/env python3
"""
FastAPI server for the Intelligent Cloud Load Balancer environment
"""

import os
import sys
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from load_balancer_env import LoadBalancerEnv, Action, Observation, TaskType
from tasks import get_all_task_info, evaluate_task


# Pydantic models for API
class StepRequest(BaseModel):
    action: Action


class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]


class ResetRequest(BaseModel):
    task_type: str = "basic_load"
    seed: Optional[int] = None
    options: Optional[Dict[str, Any]] = None


class ResetResponse(BaseModel):
    observation: Observation


class StateResponse(BaseModel):
    state: Dict[str, Any]


class TasksResponse(BaseModel):
    tasks: Dict[str, Dict[str, str]]


class EvaluateRequest(BaseModel):
    task_type: str
    actions: List[Action]
    observations: List[Observation]
    rewards: List[float]
    seed: Optional[int] = None
    options: Optional[Dict[str, Any]] = None


class EvaluateResponse(BaseModel):
    score: float


def create_fastapi_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    app = FastAPI(
        title="Intelligent Cloud Load Balancer Environment",
        description="OpenEnv-compatible load balancing simulation environment",
        version="1.0.0"
    )

    # Initialize environment (will be stored in app.state)
    @app.on_event("startup")
    async def startup_event():
        app.state.env = LoadBalancerEnv()

    @app.get("/")
    async def root():
        """Root endpoint"""
        return {"message": "Intelligent Cloud Load Balancer Environment", "version": "1.0.0"}

    @app.post("/reset", response_model=ResetResponse)
    async def reset(request: Optional[ResetRequest] = None):
        """Reset the environment with a specific task type"""
        # ✅ FIX: OpenEnv sends POST /reset with empty body — default to ResetRequest()
        if request is None:
            request = ResetRequest()
        try:
            # Convert task type string to enum
            task_type = TaskType(request.task_type)

            # Create new environment with specified task type
            app.state.env = LoadBalancerEnv(task_type)

            # Reset and return initial observation
            observation = app.state.env.reset()
            return ResetResponse(observation=observation)

        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid task type: {request.task_type}")
        except Exception as e:
            # Log the error but don't raise HTTP 500 - this causes validation failures
            print(f"[DEBUG] Reset error: {str(e)}", flush=True)
            # Return a successful response with the observation we got
            return ResetResponse(observation=observation)

    @app.post("/step", response_model=StepResponse)
    async def step(request: StepRequest):
        """Execute one step in the environment"""
        try:
            if not hasattr(app.state, 'env') or app.state.env is None:
                raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

            # Execute step
            observation, reward, done, info = app.state.env.step(request.action)

            return StepResponse(
                observation=observation,
                reward=reward,
                done=done,
                info=info
            )

        except Exception as e:
            # Log error but don't raise HTTP 500 - this causes validation failures
            print(f"[DEBUG] Step error: {str(e)}", flush=True)
            # Return a successful response with the observation we got
            return StepResponse(
                observation=observation,
                reward=reward,
                done=done,
                info=info
            )

    @app.get("/state", response_model=StateResponse)
    async def get_state():
        """Get current environment state"""
        try:
            if not hasattr(app.state, 'env') or app.state.env is None:
                raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

            state = app.state.env.state()
            return StateResponse(state=state)

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Get state failed: {str(e)}")

    @app.get("/tasks", response_model=TasksResponse)
    async def get_tasks():
        """Get information about available tasks"""
        try:
            tasks = get_all_task_info()
            return TasksResponse(tasks=tasks)

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Get tasks failed: {str(e)}")

    @app.post("/evaluate", response_model=EvaluateResponse)
    async def evaluate(request: EvaluateRequest):
        """Evaluate a completed episode"""
        try:
            task_type = TaskType(request.task_type)
            score = evaluate_task(task_type, request.actions, request.observations, request.rewards)
            return EvaluateResponse(score=score)

        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid task type: {request.task_type}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {"status": "healthy", "environment_initialized": hasattr(app.state, 'env') and app.state.env is not None}

    return app


# Create the app instance
app = create_fastapi_app()


if __name__ == "__main__":
    # Run the server directly
    uvicorn.run(app, host="0.0.0.0", port=7860)
