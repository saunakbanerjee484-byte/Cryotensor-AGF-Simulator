"""
ui_components/marquee_engine.py
==================================
The "No Black Box" Marquee: renders an HTML/CSS scrolling banner across the top
of every module screen, showing the exact differential equations and physics
logic currently active in the backend. Pure presentation -- no physics math.
"""

from __future__ import annotations
import html


_MARQUEE_CSS = """
<style>
.cryotensor-marquee-wrap {
    width: 100%;
    background: linear-gradient(90deg, #061a2b 0%, #0a2e4d 50%, #061a2b 100%);
    border: 1px solid #1c4f7c;
    border-radius: 6px;
    padding: 6px 0;
    margin-bottom: 14px;
    overflow: hidden;
    box-shadow: inset 0 0 12px rgba(0, 180, 255, 0.15);
}
.cryotensor-marquee-wrap marquee {
    color: #6be0ff;
    font-family: "Courier New", monospace;
    font-size: 15px;
    letter-spacing: 0.5px;
    text-shadow: 0 0 6px rgba(107, 224, 255, 0.6);
}
.cryotensor-marquee-label {
    display: inline-block;
    background: #ff9f1c;
    color: #061a2b;
    font-family: "Courier New", monospace;
    font-weight: 700;
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 4px;
    margin: 0 6px 4px 10px;
    letter-spacing: 1px;
}
</style>
"""


def render_marquee(equations: list[str], label: str = "LIVE BACKEND PHYSICS") -> str:
    """
    Build the HTML string for the scrolling equation banner.
    Caller is responsible for rendering it via:
        st.markdown(render_marquee([...]), unsafe_allow_html=True)

    `equations` should be plain-text / unicode math strings (e.g. using
    partial-derivative symbols) -- they are HTML-escaped internally so any
    special characters render safely.
    """
    safe_eqs = [html.escape(eq) for eq in equations]
    joined = "&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;".join(safe_eqs)
    safe_label = html.escape(label)

    return (
        _MARQUEE_CSS
        + '<div class="cryotensor-marquee-wrap">'
        + f'<span class="cryotensor-marquee-label">{safe_label}</span>'
        + f'<marquee behavior="scroll" direction="left" scrollamount="6">{joined}</marquee>'
        + "</div>"
    )


# Canonical equation sets per sub-module (single source of truth so app.py
# stays declarative and doesn't hand-author physics text inline).
MODULE_1A_EQUATIONS = [
    "Governing PDE:  ρ·c_app(T)·∂T/∂t = ∇·(k(T)·∇T)   [2D transient heat conduction]",
    "Apparent Heat Capacity:  c_app(T) = c(T) + L·d(f_l)/dT",
    "Mixing law:  k(T) = f_l·k_unfrozen + (1−f_l)·k_frozen",
]

MODULE_1B_EQUATIONS = [
    "Freeze-pipe boundary:  T(pipe) = T_coolant  (Dirichlet)",
    "Far-field boundary:  T(∂Ω_outer) = T_ground  (Dirichlet)",
    "Initial condition:  T(x,y,0) = T_ground,0",
]

MODULE_1C_EQUATIONS = [
    "Stefan condition (smoothed):  f_l(T) = clip[(T − (T_f − ΔT/2)) / ΔT , 0, 1]",
    "Discretization:  Backward-Euler + 2D 5-point Laplacian (scipy.sparse.kron)",
    "Nonlinear closure:  Picard fixed-point iteration on c_app(T), k(T) per Δt",
    "Freeze-front radius:  r_front(t) = mean |X_edge − X_pipe|  where T=0°C isotherm",
]
