"""Capture high-resolution screenshots of the PikMatrix Login and Authenticated Terminal."""

import os
import shutil
import time
import subprocess
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

ARTIFACT_DIR = Path(r"C:\Users\ZED\.gemini\antigravity-ide\brain\a177f5fa-639e-4f11-a23e-ce26d78c7b45")
PROJECT_ROOT = Path(r"c:\Users\ZED\Desktop\PikMatrix")

def run():
    port = 8899
    server_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port), "--host", "127.0.0.1"],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"Started temporary uvicorn on port {port} (pid {server_process.pid})...", flush=True)
    time.sleep(3)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                headless=True,
            )
            context = browser.new_context(viewport={"width": 1440, "height": 900})
            page = context.new_page()

            # 1. Attempt visiting / unauthenticated -> verify redirect to /login
            print("Navigating to http://127.0.0.1:8899/...")
            page.goto(f"http://127.0.0.1:{port}/", wait_until="networkidle")
            print(f"Current URL after redirect: {page.url}")
            time.sleep(1)

            # Screenshot of the Login screen
            login_img = PROJECT_ROOT / "preview_login_screen.png"
            page.screenshot(path=str(login_img))
            print(f"Saved: {login_img}")

            # 2. Switch to Register tab and take screenshot
            page.click("#tab-register-btn")
            time.sleep(0.5)
            register_img = PROJECT_ROOT / "preview_register_screen.png"
            page.screenshot(path=str(register_img))
            print(f"Saved: {register_img}")

            page.on("console", lambda msg: print("CONSOLE:", msg.text))
            page.on("response", lambda r: print("RESPONSE:", r.status, r.url, "HX-Redirect:", r.headers.get("hx-redirect")))

            # 3. Switch back to Login and fill demo credentials
            page.click("#tab-login-btn")
            time.sleep(0.5)
            page.fill("#login-email", "demo@pikmatrix.com")
            page.fill("#login-password", "pikmatrix2026")
            time.sleep(0.5)

            # Capture credentials filled
            filled_img = PROJECT_ROOT / "preview_login_filled.png"
            page.screenshot(path=str(filled_img))
            print(f"Saved: {filled_img}")

            # 4. Submit form
            print("Submitting login form...")
            page.click("#loginSubmitBtn")
            time.sleep(4)
            print("After submit, current URL is:", page.url)

            dashboard_img = PROJECT_ROOT / "preview_authenticated_dashboard.png"
            page.screenshot(path=str(dashboard_img))
            print(f"Saved: {dashboard_img}")


            # Copy all images to artifact dir
            ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
            for img in [login_img, register_img, filled_img, dashboard_img]:
                shutil.copy2(img, ARTIFACT_DIR / img.name)
                print(f"Copied {img.name} to artifact directory")

            browser.close()
    finally:
        server_process.terminate()
        try:
            server_process.wait(timeout=3)
        except Exception:
            server_process.kill()
        print("Server process stopped.")

if __name__ == "__main__":
    run()
