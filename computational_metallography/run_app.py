import os, sys, multiprocessing

def resource_base():
    # when frozen, bundled files live in PyInstaller's temp-extract dir
    return getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))

def main():
    multiprocessing.freeze_support()          # REQUIRED on Windows builds
    from streamlit.web import cli as stcli
    app = os.path.join(resource_base(), "app.py")
    sys.argv = [
        "streamlit", "run", app,
        "--global.developmentMode=false",
    ]
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()

