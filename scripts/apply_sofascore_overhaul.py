import re

file_path = r"c:\Users\ZED\Desktop\PikMatrix\templates\index.html"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update <style> block
sofascore_css = """    <style>
        :root {
            /* Sofascore Slate-Navy Design System Variables */
            --bg-page: #0f141c;
            --bg-surface: #161d27;
            --bg-surface-hover: #1f2836;
            --bg-surface-elevated: #242e3f;
            --border-subtle: #242e3f;
            --text-primary: #ffffff;
            --text-secondary: #8c9ba8;
            --text-muted: #5e6d7d;
            --accent-blue: #2372f5;
            --accent-gray: #8c9ba8;
            --accent-green: var(--accent-gray);
            --accent-red: #e02b37;
            --font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;

            /* Legacy Token Mappings */
            --font-main: var(--font-family);
            --color-bg-base: var(--bg-page);
            --color-surface: var(--bg-surface);
            --color-surface-elevated: var(--bg-surface-elevated);
            --color-border: var(--border-subtle);
            --color-text-main: var(--text-primary);
            --color-text-muted: var(--text-muted);
            --bg-base: var(--bg-page);
            --bg-card: var(--bg-surface);
            --bg-card-hover: var(--bg-surface-hover);
            --bg-nested: var(--bg-surface-elevated);
            --bg-nested-hover: var(--bg-surface-hover);
            --border-color: var(--border-subtle);
            --border-grey: var(--border-subtle);
            --border-radius: 6px;
            --color-vibrant-blue: var(--accent-blue);
            --color-gold: var(--accent-gray);
            --color-crimson-red: var(--accent-red);
            --accent-blue-hover: #1b5edb;
            --accent-blue-bg: rgba(35, 114, 245, 0.15);
            --accent-blue-border: rgba(35, 114, 245, 0.35);
        }

        * {
            box-sizing: border-box;
            box-shadow: none !important;
            text-shadow: none !important;
        }

        body {
            background-color: var(--bg-page) !important;
            color: var(--text-primary);
            font-family: var(--font-family);
            min-height: 100vh;
            margin: 0;
            overflow-x: hidden;
            -webkit-font-smoothing: antialiased;
        }

        button, input, select, textarea {
            font-family: var(--font-family);
        }

        .mono {
            font-family: 'JetBrains Mono', monospace;
        }

        /* Slim Matte Scrollbars */
        ::-webkit-scrollbar {
            width: 5px;
            height: 5px;
        }
        ::-webkit-scrollbar-track {
            background: var(--bg-page);
        }
        ::-webkit-scrollbar-thumb {
            background: var(--border-subtle);
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: var(--text-muted);
        }

        /* Top Navigation Header (52px height) */
        .app-header {
            background-color: var(--bg-surface);
            border-bottom: 1px solid var(--border-subtle);
            position: sticky;
            top: 0;
            z-index: 1040;
        }

        .header-top-tier {
            display: flex;
            align-items: center;
            justify-content: space-between;
            height: 52px;
            padding: 0 1.25rem;
            border-bottom: 1px solid var(--border-subtle);
            gap: 1.25rem;
        }

        .brand-wrapper {
            display: flex;
            align-items: center;
            gap: 8px;
            text-decoration: none;
            color: var(--text-primary);
            user-select: none;
        }

        .brand-icon-box {
            width: 26px;
            height: 26px;
            border-radius: 6px;
            background: var(--accent-blue);
            display: flex;
            align-items: center;
            justify-content: center;
            color: #ffffff;
            font-size: 0.90rem;
        }

        .brand-title {
            font-size: 1.15rem;
            font-weight: 800;
            letter-spacing: -0.3px;
            color: #ffffff;
            line-height: 1;
        }

        /* User Greeting */
        .user-greeting-box {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 3px 10px;
            background: var(--bg-page);
            border: 1px solid var(--border-subtle);
            border-radius: 6px;
        }

        .user-avatar {
            width: 22px;
            height: 22px;
            border-radius: 50%;
            background: var(--accent-blue);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.70rem;
            color: #ffffff;
        }

        .user-greeting-text {
            font-size: 0.78rem;
            font-weight: 600;
            color: var(--text-primary);
        }

        .user-tier-pill {
            font-size: 0.65rem;
            font-weight: 700;
            padding: 2px 6px;
            border-radius: 4px;
            background: var(--bg-surface-elevated);
            color: var(--text-secondary);
            border: 1px solid var(--border-subtle);
        }

        /* Search Bar */
        .search-container {
            flex: 1;
            max-width: 360px;
            position: relative;
        }

        .search-input-pill {
            width: 100%;
            background: var(--bg-page);
            border: 1px solid var(--border-subtle);
            border-radius: 6px;
            padding: 0.35rem 0.85rem 0.35rem 2rem;
            color: var(--text-primary);
            font-size: 0.80rem;
            outline: none;
            transition: border-color 0.15s ease;
        }

        .search-input-pill::placeholder {
            color: var(--text-muted);
        }

        .search-input-pill:focus {
            border-color: var(--accent-blue);
            box-shadow: none !important;
        }

        .search-icon-pos {
            position: absolute;
            left: 0.70rem;
            top: 50%;
            transform: translateY(-50%);
            color: var(--text-muted);
            font-size: 0.82rem;
            pointer-events: none;
        }

        /* Bet Slip Header Button */
        .slip-header-btn {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            background: var(--accent-blue);
            color: #ffffff;
            border: none;
            border-radius: 6px;
            padding: 0.35rem 0.85rem;
            font-size: 0.78rem;
            font-weight: 700;
            cursor: pointer;
            transition: background 0.15s ease;
        }

        .slip-header-btn:hover {
            background: var(--accent-blue-hover);
        }

        .slip-count-badge {
            background: #ffffff;
            color: var(--bg-page);
            font-size: 0.68rem;
            font-weight: 800;
            padding: 1px 6px;
            border-radius: 10px;
        }

        /* Sport Filter Strip immediately below header */
        .header-sub-tier {
            display: flex;
            align-items: center;
            justify-content: flex-start;
            height: 40px;
            padding: 0 1.25rem;
            background-color: var(--bg-surface);
            border-bottom: 1px solid var(--border-subtle);
            gap: 8px;
            overflow-x: auto;
        }

        .sports-nav-group {
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .sport-chip {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 0.25rem 0.70rem;
            border-radius: 4px;
            font-size: 0.78rem;
            font-weight: 600;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.15s ease;
            white-space: nowrap;
            border: 1px solid var(--border-subtle);
            background: var(--bg-surface-elevated);
            text-decoration: none;
            user-select: none;
        }

        .sport-chip:hover {
            color: var(--text-primary);
            background: var(--bg-surface-hover);
        }

        .sport-chip.active {
            background: var(--accent-blue) !important;
            border-color: var(--accent-blue) !important;
            color: #ffffff !important;
            font-weight: 700;
        }

        .sport-chip-count {
            font-size: 0.68rem;
            font-weight: 700;
            padding: 1px 5px;
            border-radius: 8px;
            background: rgba(0, 0, 0, 0.25);
            color: inherit;
        }

        .sport-chip.active .sport-chip-count {
            background: rgba(255, 255, 255, 0.25);
            color: #ffffff;
        }

        /* 3-Column Layout Container */
        .app-layout-container {
            display: flex;
            width: 100%;
            height: calc(100vh - 92px);
            overflow: hidden;
            background-color: var(--bg-page);
            position: relative;
        }

        /* Column 1: Left Navigation (20%) */
        .layout-col-left {
            flex: 0 0 20%;
            max-width: 20%;
            background-color: var(--bg-page);
            border-right: 1px solid var(--border-subtle);
            padding: 0.75rem;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            overflow-y: auto;
            position: relative;
            z-index: 2;
        }

        /* Column 2: Center Match Feed (40%) */
        .layout-col-center {
            flex: 0 0 40%;
            max-width: 40%;
            background-color: var(--bg-page);
            border-right: 1px solid var(--border-subtle);
            padding: 0;
            display: flex;
            flex-direction: column;
            overflow-y: auto;
            position: relative;
            z-index: 2;
        }

        /* Column 3: Right Match Detail & Analysis (40%) */
        .layout-col-right {
            flex: 0 0 40%;
            max-width: 40%;
            background-color: var(--bg-page);
            padding: 0.75rem 1rem;
            overflow-y: auto;
            position: relative;
            z-index: 2;
        }

        @media (max-width: 1199px) {
            .app-layout-container {
                flex-wrap: wrap;
                height: auto;
            }
            .layout-col-left {
                flex: 0 0 100%;
                max-width: 100%;
                border-right: none;
                border-bottom: 1px solid var(--border-subtle);
            }
            .layout-col-center {
                flex: 0 0 50%;
                max-width: 50%;
                height: 85vh;
            }
            .layout-col-right {
                flex: 0 0 50%;
                max-width: 50%;
                height: 85vh;
            }
        }

        @media (max-width: 767px) {
            .layout-col-center, .layout-col-right {
                flex: 0 0 100%;
                max-width: 100%;
                height: auto;
            }
            .header-top-tier {
                padding: 0.5rem 0.85rem;
                flex-wrap: wrap;
                height: auto;
            }
            .search-container {
                order: 3;
                max-width: 100%;
            }
        }

        /* Left Sidebar UI */
        .sidebar-section-header {
            font-size: 0.70rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            color: var(--text-muted);
            margin-bottom: 0.40rem;
            padding-left: 0.20rem;
        }

        .risk-filter-card {
            background-color: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 6px;
            padding: 0.65rem 0.75rem;
            cursor: pointer;
            transition: all 0.15s ease;
        }

        .risk-filter-card:hover {
            border-color: var(--accent-blue);
            background-color: var(--bg-surface-hover);
        }

        .risk-filter-card.active {
            border-color: var(--accent-blue);
            border-left: 3px solid var(--accent-blue);
            background-color: var(--bg-surface-hover);
        }

        .badge-risk-filter {
            background: var(--bg-surface-elevated);
            color: var(--text-primary);
            border: 1px solid var(--border-subtle);
            font-size: 0.68rem;
            font-weight: 700;
            padding: 0.18rem 0.50rem;
            border-radius: 4px;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }

        .comp-nav-list {
            display: flex;
            flex-direction: column;
            gap: 2px;
        }

        .comp-nav-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.40rem 0.60rem;
            border-radius: 6px;
            font-size: 0.80rem;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.15s ease;
            border: 1px solid transparent;
            user-select: none;
        }

        .comp-nav-item:hover {
            background-color: var(--bg-surface-hover);
            color: #ffffff;
        }

        .comp-nav-item.active {
            background-color: var(--bg-surface);
            color: #ffffff;
            font-weight: 600;
            border-color: var(--border-subtle);
        }

        .comp-flag-icon {
            font-size: 0.90rem;
            margin-right: 6px;
        }

        /* Center Column: Sofascore Subheader (Date Ribbon & Status Tabs) */
        .feed-subheader {
            position: sticky;
            top: 0;
            z-index: 20;
            background: var(--bg-surface);
            border-bottom: 1px solid var(--border-subtle);
            padding: 6px 12px;
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .date-ribbon-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 6px;
        }

        .date-ribbon-items {
            display: flex;
            align-items: center;
            gap: 4px;
            flex: 1;
            justify-content: center;
            overflow-x: auto;
        }

        .date-ribbon-arrow {
            background: transparent;
            border: none;
            color: var(--text-secondary);
            font-size: 0.78rem;
            padding: 3px 6px;
            cursor: pointer;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: color 0.15s ease, background 0.15s ease;
        }

        .date-ribbon-arrow:hover {
            color: var(--text-primary);
            background: var(--bg-surface-hover);
        }

        .date-ribbon-btn {
            background: transparent;
            border: none;
            color: var(--text-secondary);
            font-size: 0.76rem;
            font-weight: 600;
            padding: 3px 8px;
            border-radius: 4px;
            cursor: pointer;
            white-space: nowrap;
            transition: all 0.15s ease;
            user-select: none;
        }

        .date-ribbon-btn:hover {
            color: var(--text-primary);
            background: var(--bg-surface-hover);
        }

        .date-ribbon-btn.active {
            background: var(--accent-blue);
            color: #ffffff;
            font-weight: 700;
        }

        .status-tabs-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-top: 1px solid var(--border-subtle);
            padding-top: 5px;
        }

        .status-tab-btn {
            background: transparent;
            border: none;
            color: var(--text-secondary);
            font-size: 0.74rem;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.15s ease;
            user-select: none;
        }

        .status-tab-btn:hover {
            color: var(--text-primary);
            background: var(--bg-surface-hover);
        }

        .status-tab-btn.active {
            background: var(--bg-surface-elevated);
            color: var(--text-primary);
            font-weight: 700;
        }

        /* League Grouping Headers (Slim 32px bar) */
        .league-group-bar {
            height: 32px;
            background: var(--bg-page);
            padding: 4px 12px;
            border-bottom: 1px solid var(--border-subtle);
            display: flex;
            align-items: center;
            justify-content: space-between;
            user-select: none;
        }

        .league-group-title {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.76rem;
            font-weight: 700;
            color: var(--text-primary);
            text-transform: uppercase;
            letter-spacing: 0.3px;
            overflow: hidden;
            max-width: 80%;
        }

        .league-group-icon {
            font-size: 0.90rem;
            flex-shrink: 0;
        }

        .league-group-name {
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .league-group-count {
            font-size: 0.70rem;
            color: var(--text-secondary);
            font-weight: 700;
        }

        /* Sofascore-Style Slim Match Rows (~54px) */
        .sofa-match-row {
            min-height: 54px;
            height: 54px;
            background: var(--bg-surface);
            border-bottom: 1px solid var(--border-subtle);
            border-left: 3px solid transparent;
            display: flex;
            align-items: center;
            padding: 0 12px;
            cursor: pointer;
            transition: background-color 0.15s ease, border-left-color 0.15s ease;
            user-select: none;
        }

        .sofa-match-row:hover {
            background-color: var(--bg-surface-hover) !important;
        }

        .sofa-match-row.active-match {
            background-color: var(--bg-surface-hover) !important;
            border-left: 3px solid var(--accent-blue) !important;
        }

        .sofa-left-col {
            width: 50px;
            flex-shrink: 0;
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            justify-content: center;
            font-size: 0.76rem;
            font-weight: 600;
            color: var(--text-secondary);
            line-height: 1.2;
        }

        .sofa-left-col .live-minute {
            color: var(--accent-red);
            font-weight: 700;
        }

        .sofa-left-col .ft-text {
            color: var(--text-muted);
            font-weight: 700;
            font-size: 0.72rem;
        }

        .sofa-middle-col {
            flex: 1;
            min-width: 0;
            display: flex;
            flex-direction: column;
            justify-content: center;
            gap: 3px;
            padding: 0 10px;
        }

        .sofa-team-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            height: 19px;
        }

        .sofa-team-info {
            display: flex;
            align-items: center;
            gap: 7px;
            min-width: 0;
            overflow: hidden;
        }

        .sofa-team-badge {
            width: 17px;
            height: 17px;
            border-radius: 50%;
            background: var(--bg-surface-elevated);
            border: 1px solid var(--border-subtle);
            color: #ffffff;
            font-size: 0.60rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }

        .sofa-team-name {
            font-size: 0.82rem;
            font-weight: 600;
            color: var(--text-primary);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .sofa-score {
            font-size: 0.82rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-left: 8px;
            flex-shrink: 0;
        }

        .sofa-right-col {
            flex-shrink: 0;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            min-width: 78px;
        }

        .pred-chip-pill {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.72rem;
            font-weight: 700;
            white-space: nowrap;
            border: 1px solid transparent;
            line-height: 1.2;
        }

        .pred-chip-edge {
            background: rgba(140, 155, 168, 0.12);
            color: var(--accent-gray);
            border-color: rgba(140, 155, 168, 0.35);
        }

        .pred-chip-lock {
            background: rgba(35, 114, 245, 0.12);
            color: var(--accent-blue);
            border-color: rgba(35, 114, 245, 0.35);
        }

        .pred-chip-neutral {
            background: var(--bg-surface-elevated);
            color: var(--text-secondary);
            border-color: var(--border-subtle);
        }

        .pred-chip-red {
            background: rgba(224, 43, 55, 0.12);
            color: var(--accent-red);
            border-color: rgba(224, 43, 55, 0.35);
        }

        /* Terminal Badges */
        .badge-pill-tint {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 2px 7px;
            border-radius: 4px;
            font-size: 0.70rem;
            font-weight: 700;
            line-height: 1.2;
            border: 1px solid transparent;
            white-space: nowrap;
        }

        .badge-pill-tint.tint-gold {
            background: rgba(140, 155, 168, 0.12) !important;
            color: var(--accent-gray) !important;
            border-color: rgba(140, 155, 168, 0.35) !important;
        }

        .badge-pill-tint.tint-neutral {
            background: var(--bg-surface-elevated) !important;
            color: var(--text-secondary) !important;
            border-color: var(--border-subtle) !important;
        }

        .badge-pill-tint.tint-blue {
            background: rgba(35, 114, 245, 0.12) !important;
            color: var(--accent-blue) !important;
            border-color: rgba(35, 114, 245, 0.35) !important;
        }

        .badge-pill-tint.tint-red {
            background: rgba(224, 43, 55, 0.12) !important;
            color: var(--accent-red) !important;
            border-color: rgba(224, 43, 55, 0.35) !important;
        }

        /* Right Panel: Match Detail Hub */
        .right-panel-wrapper {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        /* Top Header Card (Matchup Display) */
        .sofa-header-card {
            background-color: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 12px 16px;
        }

        .sofa-header-league-row {
            text-align: center;
            font-size: 0.74rem;
            font-weight: 600;
            color: var(--text-secondary);
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
        }

        .sofa-header-matchup {
            display: flex;
            align-items: center;
            justify-content: space-around;
        }

        .sofa-matchup-team {
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            flex: 1;
            max-width: 140px;
        }

        .sofa-matchup-crest {
            width: 38px;
            height: 38px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.88rem;
            font-weight: 800;
            color: #ffffff;
            margin-bottom: 0.30rem;
            background-color: var(--bg-surface-elevated);
            border: 1px solid var(--border-subtle);
        }

        .sofa-matchup-name {
            font-size: 0.90rem;
            font-weight: 700;
            color: #ffffff;
            line-height: 1.2;
        }

        .sofa-matchup-center {
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            padding: 0 0.75rem;
        }

        .sofa-matchup-score {
            font-size: 1.55rem;
            font-weight: 800;
            color: #ffffff;
            line-height: 1.1;
        }

        .sofa-matchup-time {
            font-size: 1.30rem;
            font-weight: 800;
            color: #ffffff;
            line-height: 1.1;
        }

        .sofa-matchup-status {
            font-size: 0.68rem;
            font-weight: 700;
            color: var(--text-secondary);
            background: var(--bg-surface-elevated);
            border: 1px solid var(--border-subtle);
            padding: 2px 8px;
            border-radius: 4px;
            margin-top: 4px;
        }

        /* Prediction Cards (Replacing Sofascore's "Who Will Win" block) */
        .market-card {
            background: var(--bg-surface) !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: 8px !important;
            padding: 12px 14px;
            margin-bottom: 0.60rem;
            transition: border-color 0.15s ease;
        }

        .fortress-lock-card {
            border-left: 3px solid var(--accent-blue) !important;
        }

        .alpha-edge-card {
            border-left: 3px solid var(--accent-gray) !important;
        }

        .pred-card-header {
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }

        .pred-card-title {
            font-size: 0.95rem;
            font-weight: 700;
            color: #ffffff;
            letter-spacing: -0.1px;
            line-height: 1.25;
        }

        /* Radial Dial */
        .radial-dial-wrapper {
            position: relative;
            width: 52px;
            height: 52px;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }

        .radial-dial-svg {
            width: 52px;
            height: 52px;
            overflow: visible;
        }

        .radial-track {
            fill: none;
            stroke: var(--bg-surface-elevated);
        }

        .radial-progress {
            fill: none;
            stroke: var(--accent-blue) !important;
            stroke-linecap: round;
            filter: none !important;
            transition: stroke-dashoffset 0.4s ease;
        }

        .radial-dial-center {
            position: absolute;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
        }

        .radial-dial-num {
            font-size: 0.88rem;
            font-weight: 800;
            color: #ffffff;
            line-height: 1;
        }

        .radial-dial-label {
            font-size: 0.48rem;
            font-weight: 700;
            color: var(--accent-blue);
            letter-spacing: 0.5px;
            margin-top: 1px;
        }

        /* Alpha Edge Tag */
        .alpha-edge-tag {
            background: rgba(140, 155, 168, 0.12);
            border: 1px solid rgba(140, 155, 168, 0.35);
            color: var(--accent-gray);
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.74rem;
            font-weight: 800;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }

        /* 4-Cell EV Math Table */
        .ev-math-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 6px;
            background: var(--bg-surface-elevated);
            border: 1px solid var(--border-subtle);
            border-radius: 6px;
            padding: 6px;
            margin: 8px 0;
        }

        .ev-math-cell {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 4px 2px;
            background: var(--bg-surface);
            border-radius: 4px;
        }

        .ev-math-label {
            font-size: 0.60rem;
            font-weight: 700;
            color: var(--text-muted);
            letter-spacing: 0.4px;
            margin-bottom: 2px;
        }

        .ev-math-val {
            font-size: 0.85rem;
            font-weight: 800;
            color: #ffffff;
        }

        /* Why Accordion */
        .why-accordion-wrapper {
            margin-top: 6px;
        }

        .why-accordion-btn {
            width: 100%;
            background: var(--bg-surface-elevated);
            border: 1px solid var(--border-subtle);
            border-radius: 4px;
            padding: 5px 8px;
            color: var(--text-secondary);
            font-size: 0.74rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            justify-content: space-between;
            cursor: pointer;
            transition: all 0.15s ease;
        }

        .why-accordion-btn:hover {
            color: var(--text-primary);
            background: var(--bg-surface-hover);
        }

        .why-accordion-body {
            display: none;
            padding: 8px 10px;
            background: var(--bg-page);
            border: 1px solid var(--border-subtle);
            border-top: none;
            border-radius: 0 0 4px 4px;
            font-size: 0.75rem;
            color: var(--text-secondary);
            line-height: 1.45;
        }

        .why-accordion-body.show {
            display: block;
        }

        /* Primary Action Button */
        .btn-add-slip-wide {
            width: 100%;
            background-color: var(--accent-blue) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 6px !important;
            padding: 0.42rem 1rem;
            font-size: 0.78rem;
            font-weight: 700 !important;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
            transition: background-color 0.15s ease;
        }

        .btn-add-slip-wide:hover {
            background-color: var(--accent-blue-hover) !important;
        }

        .btn-add-slip-wide.added {
            background-color: var(--bg-surface-elevated) !important;
            color: var(--accent-gray) !important;
            border: 1px solid rgba(140, 155, 168, 0.35) !important;
        }

        /* Task 5: Sofascore-Style Tab Links */
        .sofa-tab-bar {
            display: flex;
            align-items: center;
            border-bottom: 1px solid var(--border-subtle);
            gap: 16px;
            padding: 0 4px;
            margin: 4px 0 10px 0;
            overflow-x: auto;
        }

        .sofa-tab-link {
            background: transparent;
            border: none;
            border-bottom: 2px solid transparent;
            padding: 6px 2px;
            font-size: 0.82rem;
            font-weight: 600;
            color: var(--text-secondary);
            cursor: pointer;
            transition: color 0.15s ease, border-bottom-color 0.15s ease;
            white-space: nowrap;
            user-select: none;
        }

        .sofa-tab-link:hover {
            color: var(--text-primary);
        }

        .sofa-tab-link.active {
            color: var(--text-primary);
            font-weight: 700;
            border-bottom: 2px solid var(--accent-blue);
        }

        /* Tab Content Panel Box */
        .tab-panel-box {
            background-color: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 12px 14px;
        }

        .tab-section-title {
            font-size: 0.80rem;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 0.60rem;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .meta-kv-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
        }

        .meta-kv-item {
            background: var(--bg-page);
            border: 1px solid var(--border-subtle);
            border-radius: 4px;
            padding: 0.40rem 0.55rem;
        }

        .meta-k-label {
            font-size: 0.65rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.4px;
            margin-bottom: 2px;
        }

        .meta-v-val {
            font-size: 0.82rem;
            font-weight: 700;
            color: var(--text-primary);
        }

        /* Win Probability Horizontal Bar */
        .win-prob-bar-container {
            display: flex;
            height: 6px;
            border-radius: 2px;
            overflow: hidden;
            background: var(--bg-surface-elevated);
            margin-bottom: 0.50rem;
        }

        .prob-bar-home {
            background: var(--accent-red);
            transition: width 0.4s ease;
        }

        .prob-bar-draw {
            background: rgba(255, 255, 255, 0.20);
            transition: width 0.4s ease;
        }

        .prob-bar-away {
            background: var(--accent-blue);
            transition: width 0.4s ease;
        }

        .win-prob-labels-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 0.74rem;
            font-weight: 600;
        }

        .prob-label-home {
            color: var(--accent-red);
            font-weight: 700;
        }

        .prob-label-draw {
            color: var(--text-muted);
            font-weight: 600;
        }

        .prob-label-away {
            color: var(--accent-blue);
            font-weight: 700;
        }

        /* xG dual bar */
        .xg-metrics-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.5rem;
        }

        .xg-team-box {
            display: flex;
            flex-direction: column;
        }

        .xg-team-name {
            font-size: 0.74rem;
            color: var(--text-secondary);
            font-weight: 600;
        }

        .xg-val-text {
            font-size: 1.05rem;
            font-weight: 800;
            color: #ffffff;
            line-height: 1.1;
        }

        .xg-dual-bars-row, .h2h-dual-bars-row {
            display: flex;
            align-items: center;
            gap: 6px;
            height: 6px;
        }

        .xg-left-container, .h2h-left-container {
            flex: 1;
            height: 100%;
            background: var(--bg-surface-elevated);
            border-radius: 2px;
            overflow: hidden;
            display: flex;
            justify-content: flex-end;
        }

        .xg-center-divider, .h2h-center-divider {
            width: 2px;
            height: 10px;
            background: var(--border-subtle);
            border-radius: 2px;
            flex-shrink: 0;
        }

        .xg-right-container, .h2h-right-container {
            flex: 1;
            height: 100%;
            background: var(--bg-surface-elevated);
            border-radius: 2px;
            overflow: hidden;
            display: flex;
            justify-content: flex-start;
        }

        .xg-home-fill, .h2h-home-fill {
            background: var(--accent-red) !important;
            height: 100%;
            border-radius: 2px;
            transition: width 0.4s ease;
        }

        .xg-away-fill, .h2h-away-fill {
            background: var(--accent-blue) !important;
            height: 100%;
            border-radius: 2px;
            transition: width 0.4s ease;
        }

        .scoreline-4pill-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 6px;
        }

        .scoreline-pill-item {
            background: var(--bg-page);
            border: 1px solid var(--border-subtle);
            border-radius: 4px;
            padding: 4px 6px;
            text-align: center;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 2px;
        }

        .scoreline-val {
            font-size: 0.82rem;
            color: #ffffff;
        }

        .corner-meter-card {
            background: var(--bg-page);
            border: 1px solid var(--border-subtle);
            border-radius: 6px;
            padding: 8px 10px;
        }

        .corner-meter-track {
            position: relative;
            height: 6px;
            background: var(--bg-surface-elevated);
            border-radius: 3px;
            margin: 6px 0;
        }

        .corner-meter-fill {
            height: 100%;
            background: var(--accent-blue);
            border-radius: 3px;
        }

        .corner-meter-target {
            position: absolute;
            top: -3px;
            width: 2px;
            height: 12px;
            background: #ffffff;
            border-radius: 1px;
            transform: translateX(-50%);
        }

        /* Bet Slip Drawer */
        .slip-drawer {
            position: fixed;
            top: 0;
            right: 0;
            width: 360px;
            max-width: 90vw;
            height: 100vh;
            background: var(--bg-surface);
            border-left: 1px solid var(--border-subtle);
            z-index: 1060;
            transform: translateX(100%);
            transition: transform 0.22s cubic-bezier(0.16, 1, 0.3, 1);
            display: flex;
            flex-direction: column;
        }

        .slip-drawer.open {
            transform: translateX(0);
        }

        .slip-drawer-header {
            padding: 0.85rem 1.15rem;
            border-bottom: 1px solid var(--border-subtle);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .slip-drawer-body {
            padding: 0.85rem 1.15rem;
            overflow-y: auto;
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .slip-drawer-footer {
            padding: 1rem 1.15rem;
            border-top: 1px solid var(--border-subtle);
            background-color: var(--bg-surface-elevated);
        }

        .slip-item-card {
            background: var(--bg-page);
            border: 1px solid var(--border-subtle);
            border-radius: 6px;
            padding: 0.60rem 0.70rem;
            position: relative;
        }

        .slip-item-remove {
            position: absolute;
            top: 0.45rem;
            right: 0.45rem;
            color: var(--text-muted);
            background: none;
            border: none;
            cursor: pointer;
            font-size: 0.80rem;
        }

        .slip-item-remove:hover {
            color: #ffffff;
        }

        .slip-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: rgba(0, 0, 0, 0.65);
            z-index: 1055;
            display: none;
        }

        .slip-overlay.open {
            display: block;
        }

        .empty-view-box {
            text-align: center;
            padding: 3rem 1.5rem;
            color: var(--text-muted);
        }
    </style>"""

# Replace <style>...</style>
style_pattern = re.compile(r"<style>.*?</style>", re.DOTALL)
if not style_pattern.search(content):
    print("Error: <style> not found")
else:
    content = style_pattern.sub(sofascore_css, content, count=1)
    print("Replaced <style> block successfully")

# 2. Update Top Navigation Header
header_pattern = re.compile(r'<header class="app-header">.*?</header>', re.DOTALL)
new_header = """<header class="app-header">
        <!-- Tier 1: Top Navigation Bar (52px height) -->
        <div class="header-top-tier">
            <!-- Brand -->
            <a href="/" class="brand-wrapper">
                <div class="brand-icon-box">
                    <i class="bi bi-trophy-fill"></i>
                </div>
                <div class="brand-title">PikMatrix</div>
            </a>

            <!-- User Greeting -->
            <div class="user-greeting-box d-none d-lg-flex">
                <div class="user-avatar">
                    <i class="bi bi-person-fill"></i>
                </div>
                <div class="user-greeting-text">
                    Welcome back, <span class="fw-bold">Bettor</span>
                </div>
                <span class="user-tier-pill">PRO PICKS</span>
            </div>

            <!-- Global Search Input -->
            <div class="search-container">
                <i class="bi bi-search search-icon-pos"></i>
                <input type="text" id="globalSearchInput" class="search-input-pill" placeholder="Search matches, leagues..." value="{{ search_query }}">
            </div>

            <!-- Bet Slip Header Button -->
            <button class="slip-header-btn" onclick="toggleBetSlip()">
                <i class="bi bi-receipt"></i>
                <span>Bet Slip</span>
                <span class="slip-count-badge" id="slipBadgeCount">0</span>
            </button>
        </div>

        <!-- Tier 2: Sport Filter Strip Immediately Below Header -->
        <div class="header-sub-tier">
            <div class="sports-nav-group">
                <div class="sport-chip {% if selected_sport == 'all' %}active{% endif %}" data-sport="all" onclick="setSportFilter('all')">
                    <i class="bi bi-grid-fill"></i>
                    <span>All Sports</span>
                    <span class="sport-chip-count" id="count-all">{{ total_matches }}</span>
                </div>
                <div class="sport-chip {% if selected_sport == 'football' %}active{% endif %}" data-sport="football" onclick="setSportFilter('football')">
                    <span>⚽</span>
                    <span>Football</span>
                    <span class="sport-chip-count" id="count-football">{{ sport_match_counts.football }}</span>
                </div>
                <div class="sport-chip {% if selected_sport == 'basketball' %}active{% endif %}" data-sport="basketball" onclick="setSportFilter('basketball')">
                    <span>🏀</span>
                    <span>Basketball</span>
                    <span class="sport-chip-count" id="count-basketball">{{ sport_match_counts.basketball }}</span>
                </div>
                <div class="sport-chip {% if selected_sport == 'tennis' %}active{% endif %}" data-sport="tennis" onclick="setSportFilter('tennis')">
                    <span>🎾</span>
                    <span>Tennis</span>
                    <span class="sport-chip-count" id="count-tennis">{{ sport_match_counts.tennis }}</span>
                </div>
            </div>
        </div>
    </header>"""

if not header_pattern.search(content):
    print("Error: <header> not found")
else:
    content = header_pattern.sub(new_header, content, count=1)
    print("Replaced <header> successfully")

# 3. Update Center Column Subheader
main_col_pattern = re.compile(r'<main class="layout-col-center">.*?</main>', re.DOTALL)
new_main_col = """<main class="layout-col-center">
            <!-- Sofascore Subheader: Date Toggle Ribbon & Compact Status Tabs -->
            <div class="feed-subheader">
                <div class="date-ribbon-container">
                    <button class="date-ribbon-arrow" onclick="shiftDateFilter(-1)" title="Previous Day">
                        <i class="bi bi-chevron-left"></i>
                    </button>
                    <div class="date-ribbon-items">
                        <button class="date-ribbon-btn {% if active_date_filter == 'yesterday' %}active{% endif %}" onclick="setDateFilter('yesterday')">Yesterday</button>
                        <button class="date-ribbon-btn {% if active_date_filter == 'today' %}active{% endif %}" onclick="setDateFilter('today')">Today</button>
                        <button class="date-ribbon-btn {% if active_date_filter == 'tomorrow' %}active{% endif %}" onclick="setDateFilter('tomorrow')">Tomorrow</button>
                        <button class="date-ribbon-btn {% if active_date_filter in ('upcoming', 'all') %}active{% endif %}" onclick="setDateFilter('upcoming')">All Upcoming</button>
                    </div>
                    <button class="date-ribbon-arrow" onclick="shiftDateFilter(1)" title="Next Day">
                        <i class="bi bi-chevron-right"></i>
                    </button>
                </div>

                <div class="status-tabs-container">
                    <div class="d-flex align-items-center gap-1">
                        <button class="status-tab-btn active" id="tab-status-all" onclick="setStatusFilter('all')">All</button>
                        <button class="status-tab-btn" id="tab-status-live" onclick="setStatusFilter('live')">Live</button>
                        <button class="status-tab-btn" id="tab-status-finished" onclick="setStatusFilter('finished')">Finished</button>
                        <button class="status-tab-btn" id="tab-status-upcoming" onclick="setStatusFilter('upcoming')">Upcoming</button>
                    </div>
                    <span class="small text-secondary ms-auto mono" style="font-size: 0.70rem;" id="feedCounterText">
                        {{ matches | length }} Matches
                    </span>
                </div>
            </div>

            <!-- Match Feed Container -->
            <div id="matchFeedContainer">
                <!-- Dynamically populated via JS -->
            </div>
        </main>"""

if not main_col_pattern.search(content):
    print("Error: <main> not found")
else:
    content = main_col_pattern.sub(new_main_col, content, count=1)
    print("Replaced <main> successfully")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Saved HTML with updated CSS, Header, and Center Subheader!")
