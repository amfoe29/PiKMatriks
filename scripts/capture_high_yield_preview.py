import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from playwright.sync_api import sync_playwright
from app.auth import create_access_token

ARTIFACT_DIR = Path(r"C:\Users\ZED\.gemini\antigravity-ide\brain\e3d5969b-3642-4b40-a743-471f116fbf3d")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

token = create_access_token({"sub": "1"})

def capture():
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            headless=True,
        )
        context = browser.new_context(
            viewport={"width": 1440, "height": 900},
            device_scale_factor=1.5
        )
        context.add_cookies([{
            "name": "access_token",
            "value": token,
            "domain": "127.0.0.1",
            "path": "/",
            "httpOnly": True,
            "secure": False,
            "sameSite": "Lax",
        }])

        page = context.new_page()

        # 1. Football
        print("Capturing Football...")
        page.goto("http://127.0.0.1:8000/?sport=football", wait_until="networkidle")
        time.sleep(2)
        first_row = page.locator(".sofa-match-row").first
        if first_row.count() > 0:
            first_row.click()
            time.sleep(1.5)
        page.screenshot(path=str(ARTIFACT_DIR / "preview_football_predictions.png"))
        print("Saved: preview_football_predictions.png")

        # 2. Tennis
        print("Capturing Tennis...")
        page.goto("http://127.0.0.1:8000/?sport=tennis", wait_until="networkidle")
        time.sleep(2)
        tennis_row = page.locator(".sofa-match-row").first
        if tennis_row.count() > 0:
            tennis_row.click()
            time.sleep(1.5)
        page.screenshot(path=str(ARTIFACT_DIR / "preview_tennis_predictions.png"))
        print("Saved: preview_tennis_predictions.png")

        # 3. Basketball
        print("Capturing Basketball...")
        page.goto("http://127.0.0.1:8000/?sport=basketball", wait_until="networkidle")
        time.sleep(2)
        bball_row = page.locator(".sofa-match-row").first
        if bball_row.count() > 0:
            bball_row.click()
            time.sleep(1.5)
        page.screenshot(path=str(ARTIFACT_DIR / "preview_basketball_predictions.png"))
        print("Saved: preview_basketball_predictions.png")

        # 4. Bankroll Dashboard
        print("Capturing Bankroll Dashboard...")
        page.goto("http://127.0.0.1:8000/ui/dashboard", wait_until="networkidle")
        time.sleep(2.5)
        page.screenshot(path=str(ARTIFACT_DIR / "preview_bankroll_dashboard.png"))
        print("Saved: preview_bankroll_dashboard.png")

        browser.close()
        print("Finished all captures!")

if __name__ == "__main__":
    capture()
