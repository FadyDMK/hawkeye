Hawkeye Launcher

- Camera & Court Configuration launches the config dialog in a child window.
- Video Frame Extractor and Frame Analyzer launch as separate Python processes so the launcher window stays open.
- Full video processor is currently a placeholder dialog.

Notes:
- On Windows, the launcher uses os.spawnl with the current Python interpreter.
- Ensure you run the launcher from an activated virtual environment so child processes get correct dependencies.
