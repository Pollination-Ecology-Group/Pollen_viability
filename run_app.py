import torch
import ultralytics
import sys
from streamlit.web import cli

if __name__ == '__main__':
    sys.argv = ["streamlit", "run", "app_gui.py", "--server.headless", "true", "--server.fileWatcherType", "none"]
    sys.exit(cli.main())
