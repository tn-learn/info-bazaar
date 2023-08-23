from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from celery import Celery, states
from llamapi import REDIS_BROKER_URL, REDIS_BACKEND_URL, API_HOST, API_PORT

# FastAPI app setup
app = FastAPI()

# Celery setup using the configurations from __init__.py
celery_app = Celery(
    "llamapi.worker", broker=REDIS_BROKER_URL, backend=REDIS_BACKEND_URL
)


class PredictionRequest(BaseModel):
    payload: str


@app.post("/predict/")
async def predict(prediction_request: PredictionRequest):
    payload = prediction_request.payload
    print(f"Received request with text: {payload}")

    task = celery_app.send_task("llamapi.worker.run_inference", args=[[payload]])
    print(f"Dispatched task with ID: {task.id}")

    return {"task_id": task.id}


@app.get("/results/{task_id}")
async def get_results(task_id: str):
    print(f"Fetching results for task ID: {task_id}")

    task = celery_app.AsyncResult(task_id)

    if task.state == states.PENDING:
        return {"status": "Pending"}
    elif task.state != states.SUCCESS:
        raise HTTPException(
            status_code=500, detail="Task failed, something went wrong."
        )

    return {"status": "Success", "result": task.result}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=API_HOST, port=API_PORT)
