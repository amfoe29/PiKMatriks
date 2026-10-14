import re

def update_file(path):
    with open(path, "r", encoding="utf-8") as f:
        c = f.read()

    # 1. Colors and variables
    c = c.replace("--accent-green: #00b050;", "--accent-gray: #8c9ba8;\n            --accent-green: var(--accent-gray);")
    c = c.replace("--color-gold: var(--accent-green);", "--color-gold: var(--accent-gray);")
    c = c.replace("rgba(0, 176, 80, 0.12)", "rgba(140, 155, 168, 0.12)")
    c = c.replace("rgba(0, 176, 80, 0.35)", "rgba(140, 155, 168, 0.35)")
    c = c.replace("#00b050", "#8c9ba8")

    # 2. CSS Rules
    c = c.replace("border-left: 3px solid var(--accent-green) !important;", "border-left: 3px solid var(--accent-gray) !important;")
    c = c.replace("color: var(--accent-green);", "color: var(--accent-gray);")
    c = c.replace("color: var(--accent-green) !important;", "color: var(--accent-gray) !important;")
    c = c.replace('style="color: var(--accent-green);"', 'style="color: var(--accent-gray);"')

    # 3. Specific component styling
    c = c.replace(
        '<div class="ev-math-label" style="color: var(--accent-green);">EDGE</div>',
        '<div class="ev-math-label" style="color: var(--text-secondary);">EDGE</div>'
    )
    c = c.replace(
        '<div class="ev-math-val mono" style="color: var(--accent-green);">+${edgeVal}%</div>',
        '<div class="ev-math-val mono text-white">+${edgeVal}%</div>'
    )
    c = c.replace("Flat green text pill", "Flat gray text pill")
    c = c.replace("Clean Green Edge Tag", "Clean Gray Edge Tag")

    with open(path, "w", encoding="utf-8") as f:
        f.write(c)
    print(f"Updated {path}")

update_file("templates/index.html")
update_file("scripts/apply_sofascore_js.py")
update_file("scripts/apply_sofascore_overhaul.py")
