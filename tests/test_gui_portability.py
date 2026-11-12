import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

try:
    gui = importlib.import_module("handmouse_gui")
except ModuleNotFoundError as exc:
    if exc.name not in {"tkinter", "_tkinter"}:
        raise
    gui = None


@unittest.skipIf(gui is None, "Tkinter is not installed in this test interpreter")
class GuiPortabilityTests(unittest.TestCase):
    def test_settings_save_in_user_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "user data" / "settings.json"
            with patch.object(gui, "SETTINGS_FILE", target):
                gui.save_settings({"sensitivity": 2})
                self.assertEqual(json.loads(target.read_text())["sensitivity"], 2)
                self.assertEqual(gui.load_settings()["sensitivity"], 2)
                self.assertIn("camera", gui.load_settings())

    def test_legacy_settings_migrate_without_changing_original(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "settings.json"
            legacy.write_text('{"sensitivity": 3}', encoding="utf-8")
            target = root / "user data" / "settings.json"
            with patch.object(gui, "SETTINGS_FILE", target):
                with patch.object(gui, "__file__", str(root / "handmouse_gui.py")):
                    settings = gui.load_settings()
                    self.assertEqual(settings["sensitivity"], 3)
                    gui.save_settings(settings)
            self.assertTrue(target.exists())
            self.assertEqual(legacy.read_text(), '{"sensitivity": 3}')

    def test_worker_uses_active_environment_and_module_from_any_directory(self):
        window = Mock()
        window._res_wh.return_value = ("640", "480")
        window._cam_var.get.return_value = "USB Camera"
        window._fps_var.get.return_value = "30"
        window._hand_var.get.return_value = "any"
        window._mon_index.return_value = 0
        window._sens_var.get.return_value = 1.0
        window._scroll_var.get.return_value = 2.0
        window._tracking_mode.return_value = "relative"
        window._mirror_var.get.return_value = False
        window._show_var.get.return_value = False
        window._backend_var.get.return_value = "Intel integrated GPU (OpenVINO)"
        with patch.object(sys, "executable", "C:/custom environment/Scripts/pythonw.exe"):
            with patch.object(gui.subprocess, "Popen") as spawn:
                with patch.object(gui.threading, "Thread"):
                    gui.HandmouseGUI._start(window)
        command = spawn.call_args.args[0]
        self.assertEqual(command[:6], ["C:/custom environment/Scripts/python.exe", "-X", "utf8", "-u", "-m", "handmouse_win"])
        self.assertEqual(spawn.call_args.kwargs["encoding"], "utf-8")
        self.assertIn("--no-mirror", command)
        self.assertEqual(command[command.index("--backend") + 1], "intel")
        self.assertIn("--stats", command)
        self.assertNotIn("--gpu", command)
        self.assertNotIn("cwd", spawn.call_args.kwargs)

    def test_settings_apply_without_any_camera(self):
        window = Mock()
        window._cameras = []
        window._monitors = ["0: primary"]
        gui.HandmouseGUI._apply_settings(window, {**gui.DEFAULTS, "gpu": True})
        window._cam_var.set.assert_called_once_with("")
        window._backend_var.set.assert_called_once_with("CPU (MediaPipe)")
        self.assertFalse(window._saving)

    def test_no_camera_does_not_launch_worker(self):
        window = Mock()
        window._res_wh.return_value = ("640", "480")
        window._cam_var.get.return_value = ""
        with patch.object(gui.subprocess, "Popen") as spawn:
            with patch.object(gui.messagebox, "showerror") as error:
                gui.HandmouseGUI._start(window)
        spawn.assert_not_called()
        error.assert_called_once()


if __name__ == "__main__":
    unittest.main()
