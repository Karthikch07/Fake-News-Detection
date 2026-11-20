import streamlit as st
import streamlit.components.v1 as components


def inject_glass_css() -> None:
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&family=Inter:wght@400;500;600&display=swap');
          :root {
            --bg: #0c0c0c;
            --surface: #141414;
            --surface-2: #1a1a1a;
            --line: #262626;
            --line-strong: #333;
            --text: #ececec;
            --muted: #9a9a9a;
            --faint: #6b6b6b;
            --red: #ff2d2d;
            --red-hover: #ff4b4b;
          }
          .stApp {
            background:
              radial-gradient(900px 520px at 14% 10%, rgba(255,45,45,.22), rgba(0,0,0,0) 60%),
              radial-gradient(900px 520px at 86% 16%, rgba(255,120,120,.10), rgba(0,0,0,0) 55%),
              radial-gradient(900px 520px at 60% 92%, rgba(255,45,45,.10), rgba(0,0,0,0) 60%),
              linear-gradient(180deg, #070707 0%, #070707 55%, #0b0b0b 100%);
            font-family: 'Inter', system-ui, sans-serif; }
          header[data-testid="stHeader"] { background: transparent; }
          [data-testid="stToolbar"] { opacity: .7; }
          footer { visibility: hidden; height: 0; }
          h1, h2, h3, h4 { font-family: 'Source Serif 4', Georgia, serif !important; letter-spacing: -.005em; }

          /* Widgets */
          div[data-baseweb="input"] > div,
          div[data-baseweb="textarea"] > div,
          div[data-baseweb="select"] > div {
            background: rgba(0,0,0,.35) !important;
            border: 1px solid rgba(255,255,255,.10) !important;
            border-radius: 6px !important;
          }
          div[data-baseweb="textarea"]:focus-within > div,
          div[data-baseweb="input"]:focus-within > div { border-color: rgba(255,45,45,.75) !important; box-shadow: 0 0 18px rgba(255,45,45,.20); }
          div[data-baseweb="popover"] { z-index: 999999; }

          /* ---------- Masthead ---------- */
          .masthead { padding: 4px 0 14px; border-bottom: 1px solid var(--line); margin-bottom: 22px; position: relative; }
          .masthead::after { content: ""; position: absolute; left: 0; bottom: -1px; width: 64px; height: 3px; background: var(--red); }
          .masthead .kicker { font-size: 11px; font-weight: 600; letter-spacing: .14em; text-transform: uppercase; color: var(--red); }
          .masthead h1 { font-size: 34px !important; font-weight: 700 !important; margin: 4px 0 4px !important; padding: 0 !important; color: var(--text); }
          .masthead p { margin: 0; color: var(--muted); font-size: 14px; max-width: 680px; }
          .masthead { margin-bottom: 0; }
          .ticker-wrap { overflow: hidden; white-space: nowrap; border: 1px solid rgba(255,255,255,.08); border-top: none;
            background: linear-gradient(90deg, rgba(255,45,45,.18), rgba(0,0,0,.25) 60%); margin-bottom: 22px; }
          .ticker { display: inline-block; padding: 9px 0 9px 100%; font-size: 12.5px; color: var(--muted);
            letter-spacing: .02em; animation: marquee 32s linear infinite; }
          .ticker span { color: #ff4b4b; font-weight: 700; margin: 0 6px 0 18px; }
          .ticker-wrap:hover .ticker { animation-play-state: paused; }
          @keyframes marquee { 0% { transform: translateX(0); } 100% { transform: translateX(-100%); } }

          /* ---------- Input cards ---------- */
          .st-key-intake_card, .st-key-quick_card {
            position: relative; overflow: hidden;
            border: 1px solid rgba(255,255,255,.09) !important;
            border-radius: 14px !important;
            background: linear-gradient(160deg, rgba(255,45,45,.10), rgba(255,255,255,.025) 38%, rgba(255,255,255,.015)) !important;
            box-shadow: 0 18px 44px rgba(0,0,0,.40), inset 0 1px 0 rgba(255,255,255,.06);
            padding: 18px 20px 16px !important;
          }
          .st-key-intake_card::before, .st-key-quick_card::before {
            content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
            background: linear-gradient(180deg, #ff2d2d, rgba(255,45,45,0));
          }
          .card-head { display: flex; align-items: baseline; gap: 12px; margin-bottom: 6px; }
          .card-title { font-family: 'Source Serif 4', Georgia, serif; font-size: 19px; font-weight: 600; color: var(--text); line-height: 1.2; }
          .card-sub { font-size: 13px; color: var(--muted); margin-top: 3px; }
          .card-tag {
            margin-left: auto; font-size: 11px; letter-spacing: .06em; text-transform: uppercase;
            color: #ffb3b3; white-space: nowrap; padding: 4px 9px; border-radius: 999px;
            border: 1px solid rgba(255,90,90,.30); background: rgba(255,45,45,.08);
          }
          .card-empty {
            border: 1px dashed var(--line-strong); border-radius: 4px; padding: 16px;
            text-align: center; color: var(--muted); font-size: 13.5px;
          }
          .counter { font-size: 12px; color: var(--faint); margin-top: -6px; }
          .counter b { color: var(--muted); font-weight: 500; }
          .st-key-intake_card textarea, .st-key-quick_card textarea {
            font-size: 15px !important; line-height: 1.55 !important; color: var(--text) !important;
          }
          .st-key-intake_card textarea::placeholder, .st-key-quick_card textarea::placeholder { color: var(--faint) !important; }
          .st-key-intake_card label p, .st-key-quick_card label p { font-size: 13px !important; color: var(--muted) !important; }

          /* Buttons */
          .stApp button[kind="primary"] {
            background: linear-gradient(135deg, #ff3b3b, #c81414) !important;
            border: 1px solid rgba(255,120,120,.45) !important;
            border-radius: 8px !important; font-weight: 600 !important; min-height: 42px;
            box-shadow: 0 10px 24px rgba(255,45,45,.28);
            transition: transform .12s ease, box-shadow .12s ease, filter .12s ease;
          }
          .stApp button[kind="primary"]:hover:not(:disabled) { transform: translateY(-1px); filter: brightness(1.08); box-shadow: 0 14px 30px rgba(255,45,45,.38); }
          .stApp button[kind="primary"]:disabled { opacity: .45; box-shadow: none; }
          .stApp button[kind="secondary"] { border-radius: 8px !important; border-color: rgba(255,255,255,.12) !important; background: rgba(255,255,255,.04) !important; }
          .stApp button[kind="secondary"]:hover { border-color: rgba(255,90,90,.55) !important; color: var(--text) !important; }
          .st-key-quick_card [data-testid="stButtonGroup"] button {
            border-radius: 999px !important; font-size: 12.5px !important;
            background: rgba(255,255,255,.04) !important; border: 1px solid rgba(255,255,255,.12) !important;
          }
          .st-key-quick_card [data-testid="stButtonGroup"] button:hover { border-color: rgba(255,90,90,.55) !important; }

          /* ---------- How it works ---------- */
          .steps { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); border: 1px solid rgba(255,255,255,.08); border-radius: 14px; margin-top: 6px; background: rgba(255,255,255,.03); overflow: hidden; }
          @media (max-width: 900px) { .steps { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
          @media (max-width: 520px) { .steps { grid-template-columns: 1fr; } }
          .step { padding: 16px 18px; border-right: 1px solid rgba(255,255,255,.08); }
          .step:last-child { border-right: none; }
          .step .num { font-size: 12px; font-weight: 600; color: var(--red); margin-bottom: 6px; }
          .step .t { font-weight: 600; color: var(--text); font-size: 14.5px; margin-bottom: 4px; }
          .step .d { font-size: 13px; color: var(--muted); line-height: 1.5; }
          .section-title { display: flex; align-items: center; gap: 12px; margin: 30px 0 12px; }
          .section-title h3 { margin: 0 !important; padding: 0 !important; font-size: 22px !important; }
          .section-title .line { flex: 1; height: 1px; background: linear-gradient(90deg, rgba(255,45,45,.45), rgba(255,255,255,.10) 30%, rgba(255,255,255,0)); }

          /* Verdict */
          .verdict { border: 1px solid rgba(255,255,255,.09); border-left: 4px solid var(--faint); border-radius: 10px; padding: 14px 18px; background: rgba(255,255,255,.04); }
          .verdict.fake { background: linear-gradient(135deg, rgba(255,45,45,.20), rgba(255,45,45,.04)); box-shadow: 0 10px 28px rgba(255,45,45,.15); }
          .verdict .k { font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: var(--faint); }
          .verdict .v { font-family: 'Source Serif 4', Georgia, serif; font-size: 26px; font-weight: 700; color: var(--text); margin-top: 2px; }
          .verdict.fake { border-left-color: var(--red); }
          .verdict.fake .v { color: #ff4d5e; }
          .verdict.real { border-left-color: #3fa66b; }
          .verdict.real .v { color: #6fcf97; }
          .verdict.unsure { border-left-color: #c9a227; }
          .verdict .r { font-size: 13.5px; color: #cfcfcf; margin-top: 6px; line-height: 1.5; }
          .verdict.unsure .v { color: #f0c75e; }

          /* Claim check */
          .claim { border: 1px solid rgba(255,255,255,.09); border-radius: 12px; padding: 14px 16px 8px;
            background: rgba(255,255,255,.03); margin-bottom: 12px; }
          .claim-head { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
          .claim-n { font-size: 11px; letter-spacing: .1em; text-transform: uppercase; color: var(--faint); }
          .claim-text { font-family: 'Source Serif 4', Georgia, serif; font-size: 17px; color: var(--text); line-height: 1.45; margin-bottom: 8px; }
          .badge { font-size: 11.5px; font-weight: 600; padding: 3px 10px; border-radius: 999px; border: 1px solid; }
          .badge-fake { color: #ff6b78; border-color: rgba(255,75,90,.45); background: rgba(255,45,45,.12); }
          .badge-real { color: #6fcf97; border-color: rgba(63,166,107,.45); background: rgba(63,166,107,.10); }
          .badge-unsure { color: #f0c75e; border-color: rgba(201,162,39,.45); background: rgba(201,162,39,.10); }
          .badge-none { color: var(--muted); border-color: rgba(255,255,255,.15); background: rgba(255,255,255,.04); }
          .ev { display: flex; gap: 12px; align-items: flex-start; padding: 9px 0; border-top: 1px solid rgba(255,255,255,.06); }
          .ev:first-child { border-top: none; }
          .ev-body { flex: 1; min-width: 0; }
          .ev-body a { color: var(--text) !important; text-decoration: none; font-size: 14px; line-height: 1.45; }
          .ev-body a:hover { text-decoration: underline; text-decoration-color: rgba(255,45,45,.7); }
          .ev-meta { font-size: 12px; color: var(--faint); margin-top: 2px; }
          .ev-meta b { color: var(--muted); font-weight: 500; }
          .ev-empty { font-size: 13px; color: var(--muted); padding: 6px 0 8px; }
          .tag { flex: 0 0 86px; text-align: center; font-size: 11px; font-weight: 600; padding: 3px 0; border-radius: 6px; margin-top: 2px; }
          .tag-fake { color: #ff6b78; background: rgba(255,45,45,.14); }
          .tag-real { color: #6fcf97; background: rgba(63,166,107,.14); }
          .tag-none { color: var(--muted); background: rgba(255,255,255,.06); }
          .rep { font-size: 11px; padding: 1px 7px; border-radius: 999px; border: 1px solid rgba(255,255,255,.14); color: var(--muted); }
          .rep-fact_checker { color: #8fd3ff; border-color: rgba(120,190,255,.35); }
          .rep-reputable { color: #6fcf97; border-color: rgba(63,166,107,.35); }
          .rep-satire, .rep-unreliable { color: #ff6b78; border-color: rgba(255,75,90,.4); }
          .rep-social { color: #f0c75e; border-color: rgba(201,162,39,.35); }
          .page-info { font-size: 13px; color: var(--muted); margin: 6px 0 10px; }
          .page-info b { color: var(--text); }
          .analyzed { font-size: 14.5px; line-height: 1.6; color: #d8d8d8; }
          .analyzed mark { background: rgba(255,45,45,.28); color: #fff; padding: 0 2px; border-radius: 3px; }

          .note { border: 1px solid rgba(255,255,255,.09); border-left: 3px solid rgba(255,45,45,.6); border-radius: 10px;
            background: rgba(255,255,255,.04); padding: 12px 16px; color: #cfcfcf; font-size: 14.5px; line-height: 1.55; margin-bottom: 12px; }
          .note b { color: var(--text); }
          .note code { background: var(--bg); padding: 1px 5px; border-radius: 3px; font-size: 12.5px; }
          [data-testid="stMetric"] { background: linear-gradient(160deg, rgba(255,45,45,.08), rgba(255,255,255,.03) 50%); border: 1px solid rgba(255,255,255,.09); border-radius: 12px; padding: 12px 14px; }

          /* ---------- Sidebar ---------- */
          section[data-testid="stSidebar"] {
            background:
              radial-gradient(420px 300px at 0% 0%, rgba(255,45,45,.16), rgba(0,0,0,0) 70%),
              linear-gradient(180deg, #0d0d0f 0%, #09090a 100%) !important;
            border-right: 1px solid rgba(255,255,255,.07); }
          section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-top: 1.2rem; }
          .sb-brand { padding: 2px 0 12px; border-bottom: 1px solid var(--line); margin-bottom: 4px; }
          .sb-title { font-family: 'Source Serif 4', Georgia, serif; font-weight: 700; font-size: 18px; color: var(--text); }
          .sb-sub { font-size: 12px; color: var(--faint); margin-top: 2px; }
          .sb-section { display: flex; align-items: center; gap: 8px; margin: 20px 0 8px;
            font-size: 11px; font-weight: 600; letter-spacing: .1em; text-transform: uppercase; color: var(--faint); }
          .sb-section .sb-line { flex: 1; height: 1px; background: var(--line); }
          section[data-testid="stSidebar"] .st-key-sb_workflow [role="radiogroup"] { gap: 2px; }
          section[data-testid="stSidebar"] .st-key-sb_workflow [role="radiogroup"] > label {
            width: 100%; margin: 0; padding: 7px 10px; border-radius: 3px; border-left: 2px solid transparent; }
          section[data-testid="stSidebar"] .st-key-sb_workflow [role="radiogroup"] > label:hover { background: var(--surface-2); }
          section[data-testid="stSidebar"] .st-key-sb_workflow [role="radiogroup"] > label:has(input:checked) {
            background: linear-gradient(135deg, rgba(255,45,45,.22), rgba(255,45,45,.06)); border-left-color: var(--red); }
          section[data-testid="stSidebar"] label p { font-size: 12.5px !important; color: var(--muted) !important; }
          section[data-testid="stSidebar"] [data-testid="stExpander"] details { border-radius: 4px !important; border: 1px solid var(--line) !important; }
          section[data-testid="stSidebar"] hr { display: none; }
          .sb-card { border-radius: 10px; padding: 10px 12px; border: 1px solid rgba(255,255,255,.09); background: rgba(255,255,255,.03); border-left: 3px solid var(--line-strong); }
          .sb-card.sb-ok { border-left-color: #3fa66b; }
          .sb-card.sb-err { border-left-color: var(--red); }
          .sb-card-k { font-size: 10.5px; letter-spacing: .08em; text-transform: uppercase; color: var(--faint); }
          .sb-card-v { font-size: 13px; color: var(--text); margin-top: 3px; line-height: 1.4; }
          .sb-kpis { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-top: 10px; }
          .sb-kpis > div { border-radius: 3px; padding: 6px 8px; background: var(--bg); border: 1px solid var(--line); }
          .sb-kpis span { display: block; font-size: 10.5px; color: var(--faint); }
          .sb-kpis b { font-size: 15px; color: var(--text); font-weight: 600; }
          .sb-status { display: flex; gap: 10px; align-items: flex-start; padding: 5px 0; }
          .sb-status .dot { width: 7px; height: 7px; border-radius: 99px; margin-top: 6px; flex: 0 0 7px; }
          .sb-status .dot.on { background: #3fa66b; }
          .sb-status .dot.off { background: #444; }
          .sb-status .n { font-size: 13px; color: var(--text); font-weight: 500; }
          .sb-status .d { font-size: 11.5px; color: var(--faint); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_3d_hero(height_px: int = 240) -> None:
    """
    Renders an interactive Three.js hero scene.
    This stays inside its own container (Streamlit iframe-safe).
    """
    height_px = int(max(160, min(520, height_px)))
    components.html(
        f"""
        <div style="width:100%; height:{height_px}px; border-radius:18px; overflow:hidden;
                    border:1px solid rgba(255,255,255,.10);
                    background: rgba(255,255,255,.03);
                    box-shadow: 0 18px 40px rgba(0,0,0,.35);">
          <canvas id="c" style="display:block; width:100%; height:100%;"></canvas>
        </div>
        <script src="https://unpkg.com/three@0.160.0/build/three.min.js"></script>
        <script>
          const canvas = document.getElementById('c');
          const renderer = new THREE.WebGLRenderer({{canvas, antialias:true, alpha:true}});
          renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
          renderer.setClearColor(0x000000, 0);

          const scene = new THREE.Scene();
          const camera = new THREE.PerspectiveCamera(48, 1, 0.1, 100);
          camera.position.set(0, 0.2, 4.2);

          const resize = () => {{
            const w = canvas.clientWidth, h = canvas.clientHeight;
            renderer.setSize(w, h, false);
            camera.aspect = w / h;
            camera.updateProjectionMatrix();
          }};
          new ResizeObserver(resize).observe(canvas);
          resize();

          // Lights
          const key = new THREE.DirectionalLight(0x7c5cff, 1.25);
          key.position.set(3, 2, 2);
          scene.add(key);
          const fill = new THREE.DirectionalLight(0x12e6c4, 0.8);
          fill.position.set(-3, -1, 2);
          scene.add(fill);
          scene.add(new THREE.AmbientLight(0xffffff, 0.55));

          // Geometry
          const knotGeo = new THREE.TorusKnotGeometry(1.05, 0.34, 220, 14);
          const knotMat = new THREE.MeshPhysicalMaterial({{
            color: 0xffffff,
            metalness: 0.3,
            roughness: 0.28,
            transmission: 1.0,
            thickness: 1.2,
            ior: 1.25,
            clearcoat: 1.0,
            clearcoatRoughness: 0.25,
            emissive: 0x110a3a,
            emissiveIntensity: 0.35,
          }});
          const knot = new THREE.Mesh(knotGeo, knotMat);
          knot.rotation.set(0.4, 0.1, 0.0);
          scene.add(knot);

          const wire = new THREE.LineSegments(
            new THREE.WireframeGeometry(knotGeo),
            new THREE.LineBasicMaterial({{color: 0x7c5cff, transparent:true, opacity:0.22}})
          );
          scene.add(wire);

          // Starfield
          const starsN = 800;
          const starGeo = new THREE.BufferGeometry();
          const starPos = new Float32Array(starsN * 3);
          for (let i = 0; i < starsN; i++) {{
            const r = 6 * Math.random() + 2.2;
            const t = Math.random() * Math.PI * 2;
            const p = (Math.random() - 0.5) * 1.6;
            starPos[i*3 + 0] = Math.cos(t) * r;
            starPos[i*3 + 1] = p;
            starPos[i*3 + 2] = Math.sin(t) * r;
          }}
          starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
          const starMat = new THREE.PointsMaterial({{
            color: 0xffffff,
            size: 0.02,
            transparent: true,
            opacity: 0.65
          }});
          const stars = new THREE.Points(starGeo, starMat);
          scene.add(stars);

          // Interaction: subtle parallax on mouse move
          let targetX = 0, targetY = 0;
          const onMove = (e) => {{
            const rect = canvas.getBoundingClientRect();
            const mx = ((e.clientX - rect.left) / rect.width) * 2 - 1;
            const my = ((e.clientY - rect.top) / rect.height) * 2 - 1;
            targetX = mx * 0.35;
            targetY = -my * 0.20;
          }};
          canvas.addEventListener('mousemove', onMove);
          canvas.addEventListener('touchmove', (e) => {{
            if (!e.touches || e.touches.length === 0) return;
            onMove(e.touches[0]);
          }}, {{passive:true}});

          let t0 = performance.now();
          const animate = (t) => {{
            const dt = Math.min(0.05, (t - t0) / 1000);
            t0 = t;

            // Motion
            knot.rotation.y += dt * 0.55;
            knot.rotation.x += dt * 0.18;
            wire.rotation.copy(knot.rotation);

            camera.position.x += (targetX - camera.position.x) * 0.06;
            camera.position.y += (targetY - camera.position.y) * 0.06;
            camera.lookAt(0, 0, 0);

            stars.rotation.y += dt * 0.04;
            renderer.render(scene, camera);
            requestAnimationFrame(animate);
          }};
          requestAnimationFrame(animate);
        </script>
        """,
        height=height_px + 8,
    )

