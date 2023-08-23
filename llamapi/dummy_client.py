import requests
import time
import threading
import random

BASE_URL = "http://localhost:8000"
NUM_REQUESTS = 10  # Number of requests
MAX_SLEEP_INTERVAL = 5  # Maximum sleep interval in seconds between requests


def send_request():
    # Send a request to the /predict/ endpoint
    response = requests.post(
        f"{BASE_URL}/predict/", json={"payload": "Translate this sentence."}
    )
    data = response.json()

    task_id = data.get("task_id")

    if not task_id:
        print("Failed to start the task.")
        return

    print(f"Task ID: {task_id}")

    # Poll the /results/{task_id} endpoint to get results
    while True:
        result_response = requests.get(f"{BASE_URL}/results/{task_id}")
        result_data = result_response.json()

        status = result_data.get("status")
        if status == "Success":
            print("Result:", result_data.get("result"))
            break
        elif status == "Pending":
            print(f"Task ID {task_id}: Still pending...")
            time.sleep(2)  # Wait for 2 seconds before polling again
        else:
            print("Failed to get results.")
            break


# Start threads to send requests with random sleep intervals
threads = []
for _ in range(NUM_REQUESTS):
    t = threading.Thread(target=send_request)
    t.start()
    threads.append(t)

    # Sleep for a random duration before starting the next request
    time.sleep(random.uniform(0, MAX_SLEEP_INTERVAL))

# Wait for all threads to complete
for t in threads:
    t.join()
