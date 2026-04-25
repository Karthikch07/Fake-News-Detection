import streamlit as st
import streamlit.components.v1 as components


def inject_glass_css() -> None:
    st.markdown(
        """
        <style>
          /* App canvas */
          .stApp {
            background:
              radial-gradient(900px 520px at 14% 10%, rgba(255, 45, 45, .22), rgba(0,0,0,0) 60%),
              radial-gradient(900px 520px at 86% 16%, rgba(255, 120, 120, .10), rgba(0,0,0,0) 55%),
              radial-gradient(900px 520px at 60% 92%, rgba(255, 45, 45, .10), rgba(0,0,0,0) 60%),
              linear-gradient(180deg, #070707 0%, #070707 55%, #0b0b0b 100%);
          }

          /* Reduce default chrome */
          header[data-testid="stHeader"] { background: rgba(0,0,0,0); }
          /* Keep toolbar available so the sidebar toggle is always accessible. */
          [data-testid="stToolbar"] {
            opacity: 0.85;
            visibility: visible;
            height: auto;
          }
          footer { visibility: hidden; height: 0; }

          /* Glass helpers */
          .glass {
            border: 1px solid rgba(255,255,255,.10);
            background: rgba(255,255,255,.05);
            box-shadow:
              0 10px 30px rgba(0,0,0,.35),
              inset 0 1px 0 rgba(255,255,255,.06);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border-radius: 18px;
            padding: 18px 18px;
          }
          .glass-tight { padding: 14px 14px; border-radius: 16px; }
          .muted { color: rgba(231, 234, 243, .70); }
          .pill {
            display: inline-flex;
            gap: .45rem;
            align-items: center;
            padding: 6px 10px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,.14);
            background: linear-gradient(135deg, rgba(255,45,45,.20), rgba(255,255,255,.03));
            font-size: 12px;
            color: rgba(242, 242, 242, .90);
          }
          .kpi {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 12px;
            margin-top: 10px;
          }
          .kpi > div {
            border: 1px solid rgba(255,255,255,.10);
            background: rgba(255,255,255,.04);
            border-radius: 14px;
            padding: 10px 12px;
          }
          .kpi .label { font-size: 11px; color: rgba(231,234,243,.65); }
          .kpi .value { font-size: 15px; font-weight: 650; color: rgba(231,234,243,.92); }

          /* Make Streamlit widgets blend */
          div[data-baseweb="input"] > div,
          div[data-baseweb="textarea"] > div,
          div[data-baseweb="select"] > div {
            background: rgba(255,255,255,.04) !important;
            border: 1px solid rgba(255,255,255,.10) !important;
          }
          div[data-baseweb="popover"] { z-index: 999999; }
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

