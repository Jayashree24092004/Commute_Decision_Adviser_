"""
server/app.py
=============
Required by OpenEnv validator.
Must have a main() function and if __name__ == '__main__' block.
"""
import uvicorn
from server.main import app


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()