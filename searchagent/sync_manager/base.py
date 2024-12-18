import asyncio

from searchagent.sync_manager.db import insert_data
from searchagent.sync_manager.file_change_detector import FileChangeEventList, Watcher

INTERVAL = 3  # 5 MINS


class Monitor:
    def __init__(self, input_dir: str):
        self.input_dir = input_dir
        self.event_list: FileChangeEventList = []

    async def periodic_insert(self, event_list: FileChangeEventList):
        while True:
            await asyncio.sleep(INTERVAL)
            if event_list:
                insert_data(event_list)
                event_list.clear()

    async def run(self):
        # TODO: Periodically delete old entries
        # TODO: Allow more than one dir
        event_list: FileChangeEventList = []

        w = Watcher(self.input_dir, event_list)

        try:
            # Start the periodic insertion as an asyncio task
            asyncio.create_task(self.periodic_insert(event_list))
            await asyncio.to_thread(w.run)

            # Keep the main program running
            while True:
                await asyncio.sleep(1)  # This keeps the main thread alive
        finally:
            w.observer.stop()
            w.observer.join()
