import asyncio



from datetime import datetime, timedelta
import time
import asyncio

queue = asyncio.Future()
existing_timestamps_queue = []
lock = asyncio.Lock()


async def rate_limiter(requests_per_60):
    global existing_timestamps_queue

    while True:
        async with lock:
            new_timestamp = datetime.now()

            relevant_timestamps_queue = [timestamp for timestamp in existing_timestamps_queue if (new_timestamp - timestamp) <= timedelta(seconds=60)]

            ## then need to check timepstamps if they exist
            if len(relevant_timestamps_queue) < requests_per_60:
                existing_timestamps_queue.append(new_timestamp)
                return True
            else:
                ## must wait
                print(f"  Max requests made for free model - must wait for slot to free up..  ")
        
        await asyncio.sleep(1)
                






## need to return the queue, so it can be resolved after LLM call is called, to release the queue with set_result(True)