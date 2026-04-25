import streamlit as st
import streamlit.components.v1 as components


def render_news_alert_header() -> None:
    """
    A lightweight 'news alert system' header: siren glow + ticker + status chips.
    Uses pure HTML/CSS (no external JS libs).
    """
    components.html(
        """
        <div style="
          width: 100%;
          border-radius: 18px;
          overflow: hidden;
          border: 1px solid rgba(255,255,255,.10);
          background: linear-gradient(135deg, rgba(255,45,45,.26), rgba(255,45,45,.10) 55%, rgba(0,0,0,.10));
          box-shadow: 0 18px 40px rgba(0,0,0,.35);
        ">
          <div style="padding: 16px 18px 10px 18px; display:flex; gap:14px; align-items:center;">
            <div style="
              width: 12px; height: 12px; border-radius: 999px;
              background: #ff4b4b;
              box-shadow: 0 0 22px rgba(255,75,75,.9), 0 0 48px rgba(255,75,75,.35);
              animation: pulse 1.2s ease-in-out infinite;
            "></div>
            <div style="display:flex; flex-direction:column; gap:2px;">
              <div style="font-weight: 800; letter-spacing:.05em; font-size: 14px;">
                NEWS ALERT SYSTEM
              </div>
              <div style="opacity:.78; font-size: 12px;">
                Intake → Classification → Verification → Community Notes
              </div>
            </div>
            <div style="margin-left:auto; display:flex; gap:8px; flex-wrap:wrap;">
              <span class="chip">LIVE</span>
              <span class="chip">EVIDENCE</span>
              <span class="chip">RISK SCORE</span>
            </div>
          </div>

          <div class="ticker-wrap">
            <div class="ticker">
              <span>BREAKING:</span> Verify claims against credible sources •
              <span>ALERT:</span> Watch for sensational language & missing citations •
              <span>TIP:</span> Use URL extraction for full context •
              <span>NOTE:</span> OCR requires Tesseract on Windows •
              <span>STATUS:</span> Model options depend on installed libraries •
            </div>
          </div>
        </div>

        <style>
          @keyframes pulse {
            0% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.2); opacity: .75; }
            100% { transform: scale(1); opacity: 1; }
          }
          .chip {
            display:inline-flex;
            align-items:center;
            padding: 6px 10px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,.12);
            background: rgba(0,0,0,.18);
            font-size: 11px;
            letter-spacing:.04em;
            color: rgba(231,234,243,.92);
          }
          .ticker-wrap {
            border-top: 1px solid rgba(255,255,255,.10);
            background: rgba(0,0,0,.22);
            overflow: hidden;
            white-space: nowrap;
          }
          .ticker {
            display: inline-block;
            padding: 10px 0;
            color: rgba(231,234,243,.88);
            font-size: 12px;
            letter-spacing: .03em;
            animation: marquee 18s linear infinite;
          }
          .ticker span {
            color: #ff4b4b;
            font-weight: 800;
            margin: 0 6px 0 18px;
          }
          @keyframes marquee {
            0% { transform: translateX(100%); }
            100% { transform: translateX(-100%); }
          }
        </style>
        """,
        height=140,
    )


def sidebar_status_block(*, title: str, lines: list[str]) -> None:
    st.sidebar.markdown(f"### {title}")
    for line in lines:
        st.sidebar.write(line)

