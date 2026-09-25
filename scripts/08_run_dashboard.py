import sys
import webbrowser
from pathlib import Path
from threading import Timer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn

HOST = "127.0.0.1"
PORT = 8000


def main():
    url = f"http://{HOST}:{PORT}"
    print(f"Dashboard starting on {url}")
    print("Press Ctrl+C to stop")
    Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run("src.api:app", host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    main()
