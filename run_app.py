import os
import runpy
import sys

# If invoked directly via 'python run_app.py' in a standard shell:
if __name__ == '__main__' and len(sys.argv) > 0 and sys.argv[0].endswith("run_app.py") and "streamlit" not in os.path.basename(sys.argv[0]):
    try:
        from streamlit.web import cli
        sys.argv = ["streamlit", "run", "app_gui.py", "--server.headless", "true", "--server.fileWatcherType", "none"]
        sys.exit(cli.main())
    except Exception:
        pass

# If invoked by Streamlit Cloud (e.g. 'streamlit run run_app.py'):
gui_path = os.path.join(os.path.dirname(__file__), "app_gui.py")
runpy.run_path(gui_path, run_name="__main__")
