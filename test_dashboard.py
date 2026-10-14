import urllib.request

# 1. Test GET /ui/dashboard with HX-Request header
req_htmx = urllib.request.Request(
    "http://127.0.0.1:8000/ui/dashboard",
    headers={"HX-Request": "true"}
)
with urllib.request.urlopen(req_htmx) as resp:
    html_htmx = resp.read().decode("utf-8")
    print("HTMX /ui/dashboard Status:", resp.status)
    print("Length:", len(html_htmx))
    print("Contains bankrollYieldDashboard:", "bankrollYieldDashboard" in html_htmx)
    print("Contains yieldCurveChart canvas:", "yieldCurveChart" in html_htmx)
    print("Contains allocationChart canvas:", "allocationChart" in html_htmx)
    print("Contains net_units (+34.2 U):", "+34.2 U" in html_htmx)
    print("Contains win_rate (68%):", "68%" in html_htmx)
    print("Contains Chart.js script tag:", "chart.js" in html_htmx)

# 2. Test direct GET /ui/dashboard (Full page fallback)
with urllib.request.urlopen("http://127.0.0.1:8000/ui/dashboard") as resp:
    html_full = resp.read().decode("utf-8")
    print("\nFull page /ui/dashboard Status:", resp.status)
    print("Contains <html:", "<html" in html_full)
    print("Contains bankrollYieldDashboard:", "bankrollYieldDashboard" in html_full)

# 3. Test GET /ui/feed
req_feed = urllib.request.Request(
    "http://127.0.0.1:8000/ui/feed",
    headers={"HX-Request": "true"}
)
with urllib.request.urlopen(req_feed) as resp:
    html_feed = resp.read().decode("utf-8")
    print("\nFeed partial /ui/feed Status:", resp.status)
    print("Contains matchFeedContainer:", "matchFeedContainer" in html_feed)

# 4. Test GET /
with urllib.request.urlopen("http://127.0.0.1:8000/") as resp:
    html_home = resp.read().decode("utf-8")
    print("\nGET / Status:", resp.status)
    print("Contains hx-get=\"/ui/dashboard\":", 'hx-get="/ui/dashboard"' in html_home)
    print("Contains hx-target=\"#main-workspace\":", ('hx-target="#main-workspace"' in html_home or 'hx-target="main"' in html_home))
    print("Contains hx-swap=\"innerHTML\":", 'hx-swap="innerHTML"' in html_home)
    print("Contains hx-push-url=\"true\":", 'hx-push-url="true"' in html_home)
