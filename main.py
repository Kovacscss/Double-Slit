#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   DUPLA FENDA — Simulação Quântica TDSE 2D                                  ║
║   Gera: double_slit_full.html  (self-contained, sem dependências externas)  ║
║                                                                              ║
║   Uso: python3 run.py                                                        ║
║   Deps: pip install numpy scipy matplotlib rich Pillow                       ║
║                                                                              ║
║   Física: Split-Operator Fourier | Von Neumann | Decoerência | Wheeler      ║
║   Saída:  HTML com 3 abas — Simulação | Física | Interpretações              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ─── stdlib ───────────────────────────────────────────────────────────────────
import os, sys, time, json, base64, io, warnings
warnings.filterwarnings("ignore")

# ─── science ──────────────────────────────────────────────────────────────────
import numpy as np
from scipy.ndimage import gaussian_filter

# ─── plotting ─────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

# ─── terminal UI ──────────────────────────────────────────────────────────────
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule
from rich.text import Text
from rich import box

console = Console()

# ══════════════════════════════════════════════════════════════════════════════
#  §1  PARÂMETROS
# ══════════════════════════════════════════════════════════════════════════════
class P:
    # Grade 2D
    Nx, Ny   = 320, 160
    Lx, Ly   = 36.0, 18.0
    dt       = 0.007
    # Pacote inicial
    x0, y0   = 4.2, 0.0
    sx, sy   = 1.3, 2.0
    k0       = 6.0
    # Geometria da fenda
    x_wall_f = 0.40
    slit_a   = 0.70    # largura em λ
    slit_d   = 2.70    # separação em λ
    V0       = 5500.0
    wall_t   = 3
    # Evolução
    Nt_pre   = 240
    Nt_post  = 420
    # Renderização
    N_FRAMES = 85

    @property
    def dx(self): return self.Lx / self.Nx
    @property
    def dy(self): return self.Ly / self.Ny
    @property
    def lam(self): return 2 * np.pi / self.k0
    @property
    def slit_width(self): return self.slit_a * self.lam
    @property
    def slit_sep(self): return self.slit_d * self.lam

p = P()

# ══════════════════════════════════════════════════════════════════════════════
#  §2  FÍSICA — GRADE, BARREIRA, PROPAGADORES
# ══════════════════════════════════════════════════════════════════════════════

def build_grids():
    x  = np.linspace(0, p.Lx, p.Nx, endpoint=False)
    y  = np.linspace(-p.Ly/2, p.Ly/2, p.Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    kx = np.fft.fftfreq(p.Nx, d=p.dx) * 2 * np.pi
    ky = np.fft.fftfreq(p.Ny, d=p.dy) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    return x, y, X, Y, KX, KY


def gaussian_packet(X, Y):
    """Ψ(x,y,0) = A · exp[−(x−x₀)²/2σx²] · exp[−(y−y₀)²/2σy²] · exp(ik₀x)"""
    env = (np.exp(-((X - p.x0)**2) / (2 * p.sx**2)) *
           np.exp(-((Y - p.y0)**2) / (2 * p.sy**2)))
    psi = env * np.exp(1j * p.k0 * X)
    psi /= np.sqrt(np.sum(np.abs(psi)**2) * p.dx * p.dy)
    return psi.astype(np.complex128)


def build_barrier(y):
    """
    V(x,y) = V₀ em toda a parede, exceto nas duas fendas (V=0).
    Retorna: V, absorber_PML, x_wall, ix_wall, slit1=(ymin,ymax), slit2
    """
    V  = np.zeros((p.Nx, p.Ny))
    xw = p.x_wall_f * p.Lx
    iw = int(xw / p.dx)
    h  = p.slit_sep / 2
    hw = p.slit_width / 2
    slit1 = (+h - hw, +h + hw)   # fenda superior
    slit2 = (-h - hw, -h + hw)   # fenda inferior

    for i in range(iw - p.wall_t//2, iw + p.wall_t//2 + 1):
        if 0 <= i < p.Nx:
            V[i, :] = p.V0
            V[i, (y >= slit1[0]) & (y <= slit1[1])] = 0.0
            V[i, (y >= slit2[0]) & (y <= slit2[1])] = 0.0

    # Perfectly Matched Layer nas bordas (evita reflexões periódicas da FFT)
    ab = np.zeros((p.Nx, p.Ny))
    g = 16.0; bf = 0.08
    bx = int(bf * p.Nx); by = int(bf * p.Ny)
    rx = np.zeros(p.Nx)
    rx[:bx]  = g * (np.arange(bx, 0, -1) / bx)**2
    rx[-bx:] = g * (np.arange(1, bx+1) / bx)**2
    ry = np.zeros(p.Ny)
    ry[:by]  = g * (np.arange(by, 0, -1) / by)**2
    ry[-by:] = g * (np.arange(1, by+1) / by)**2
    RX, RY = np.meshgrid(rx, ry, indexing='ij')
    ab = np.maximum(RX, RY)

    return V, ab, xw, iw, slit1, slit2


def build_propagators(KX, KY, V, ab):
    """
    Pré-calcula os operadores do Split-Operator:
      eT  = exp(−i·k²/2·dt)       [espaço k, passo cinético completo]
      eV  = exp(−i·V·dt/2)        [espaço real, meio passo de potencial]
      eAb = exp(−ab·dt/2)         [absorvedor de borda, real]
    """
    eT  = np.exp(-1j * 0.5 * (KX**2 + KY**2) * p.dt)
    eV  = np.exp(-1j * V * p.dt / 2)
    eAb = np.exp(-ab * p.dt / 2)
    return eT, eV, eAb


def so_step(psi, eT, eV, eAb):
    """Um passo Û(dt) ≈ e^{-iVdt/2}·FFT·e^{-iTdt}·iFFT·e^{-iVdt/2}"""
    psi *= eV * eAb
    psi  = np.fft.ifft2(np.fft.fft2(psi) * eT)
    psi *= eV * eAb
    return psi


def von_neumann_project(psi, X, Y, slit, xw):
    """
    Projeção de Von Neumann na fenda superior:
      P̂₁|Ψ⟩ → máscara(F1) · Ψ, depois renormaliza.
    Retorna (psi_colapsado, probabilidade_detecção).
    """
    mask = ((Y >= slit[0]) & (Y <= slit[1]) & (X >= xw * 0.80)).astype(float)
    psi2 = psi * mask
    n = float(np.real(np.sum(np.abs(psi2)**2) * p.dx * p.dy))
    if n > 1e-12:
        psi2 /= np.sqrt(n)
    return psi2, n


def apply_decoherence(psi, strength):
    """
    Decoerência: ruído de fase gaussiano correlacionado.
    Suprime os termos de interferência off-diagonal da matriz densidade.
    """
    if strength < 1e-6:
        return psi
    raw = np.random.randn(p.Nx, p.Ny)
    smooth = gaussian_filter(raw, sigma=[max(1, int(p.Ny*0.08))] * 2)
    smooth /= (np.std(smooth) + 1e-12)
    return psi * np.exp(1j * strength * np.pi * smooth)


def delayed_choice_erase(psi, X, Y, xw, eraser=True):
    """
    Experimento de Wheeler: após a fenda, decide-se apagar ou não a informação.
    eraser=True → restaura coerência → interferência ressurge.
    """
    if eraser:
        m1 = ((Y >  0) & (X > xw * 0.5)).astype(float)
        m2 = ((Y <= 0) & (X > xw * 0.5)).astype(float)
        psi_new = (psi * m1 + psi * m2) / np.sqrt(2.0)
        n = float(np.real(np.sum(np.abs(psi_new)**2) * p.dx * p.dy))
        if n > 1e-12:
            psi_new /= np.sqrt(n)
        return psi_new
    return psi


def screen_profile(psi, smooth=2.0):
    """Intensidade integrada nos últimos 8% em x (a 'tela')."""
    last = max(1, int(p.Nx * 0.08))
    prof = np.sum(np.abs(psi[-last:])**2, axis=0)
    if smooth > 0:
        prof = gaussian_filter(prof, sigma=smooth)
    if prof.max() > 0:
        prof /= prof.max()
    return prof

# ══════════════════════════════════════════════════════════════════════════════
#  §3  RENDERER — frames PNG → base64
# ══════════════════════════════════════════════════════════════════════════════

CMAP_A  = mcolors.LinearSegmentedColormap.from_list("qA", [
    "#000000","#020a1e","#061850","#0d3a8a","#0080c0","#00c8f0","#80e8ff","#ffffff"])
CMAP_B  = mcolors.LinearSegmentedColormap.from_list("qB", [
    "#000000","#1a0010","#5a0028","#aa1000","#e04000","#ff8800","#ffe060","#ffffff"])
CMAP_D  = mcolors.LinearSegmentedColormap.from_list("qD", [
    "#000000","#001a10","#004030","#008060","#00c890","#80ffc0","#e0fff0","#ffffff"])
CMAP_DC = mcolors.LinearSegmentedColormap.from_list("qDC",[
    "#000000","#100020","#300060","#6000a0","#9030d0","#c070ff","#e8b0ff","#ffffff"])
CMAPS   = {"A": CMAP_A, "B": CMAP_B, "D": CMAP_D, "DC": CMAP_DC}


def fig_to_b64(fig, dpi=88):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi,
                bbox_inches='tight', pad_inches=0, facecolor='#000000')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')


def render_frame(prob, iw, slit1, slit2, xw,
                 cmap_key="A", detector_on=False,
                 detector_flash=False, t_val=0.0,
                 decoherence=0.0):
    cmap = CMAPS.get(cmap_key, CMAP_A)
    fig, ax = plt.subplots(figsize=(8.5, 4.2), facecolor='#000000')
    ax.set_facecolor('#000000')
    ax.set_position([0, 0, 1, 1])
    ax.axis('off')

    # ── Heatmap |Ψ|² ──────────────────────────────────────────────────────────
    display = np.power(np.clip(prob, 0, None), 0.40).T
    vmax = float(np.percentile(display[display > 0], 98)) if display.max() > 0 else 1.0
    ax.imshow(display, origin='lower', aspect='auto',
              extent=[0, p.Lx, -p.Ly/2, p.Ly/2],
              cmap=cmap, interpolation='bilinear',
              vmin=0, vmax=max(vmax, 1e-9))

    # ── Grade sutil ────────────────────────────────────────────────────────────
    for gx in np.linspace(0, p.Lx, 9)[1:-1]:
        ax.axvline(gx, color='#0a2040', lw=0.4, alpha=0.5)
    for gy in np.linspace(-p.Ly/2, p.Ly/2, 7)[1:-1]:
        ax.axhline(gy, color='#0a2040', lw=0.4, alpha=0.5)

    # ── Anteparo com fendas ────────────────────────────────────────────────────
    _draw_barrier(ax, xw, slit1, slit2)

    # ── Detector ───────────────────────────────────────────────────────────────
    if detector_on:
        _draw_detector(ax, xw, slit1, slit2, detector_flash, t_val)

    # ── Anteparo final (tela) ──────────────────────────────────────────────────
    _draw_screen(ax, prob)

    # ── Indicador de decoerência ───────────────────────────────────────────────
    if decoherence > 0.05:
        alpha = min(0.22, decoherence * 0.25)
        for _ in range(5):
            y0 = np.random.uniform(-p.Ly/2, p.Ly/2)
            ax.axhline(y0, color='#00ff88', lw=0.3, alpha=alpha * 0.5)
        ax.text(p.Lx*0.62, p.Ly/2 - p.Ly*0.08,
                f'Γ={decoherence:.2f}', color='#00ff88',
                fontsize=6, fontfamily='monospace', alpha=0.65)

    return fig_to_b64(fig)


def _draw_barrier(ax, xw, slit1, slit2):
    w = p.Lx * 0.018
    fc, ec = '#0a1a30', '#1a4a70'
    # Seção superior (acima da fenda 1)
    top = p.Ly/2 - slit1[1]
    if top > 0:
        ax.add_patch(mpatches.Rectangle(
            (xw - w/2, slit1[1]), w, top, facecolor=fc, edgecolor=ec, lw=0.8))
    # Seção entre fendas
    mid = slit1[0] - slit2[1]
    if mid > 0:
        ax.add_patch(mpatches.Rectangle(
            (xw - w/2, slit2[1]), w, mid, facecolor=fc, edgecolor=ec, lw=0.8))
    # Seção inferior (abaixo da fenda 2)
    bot = slit2[0] + p.Ly/2
    if bot > 0:
        ax.add_patch(mpatches.Rectangle(
            (xw - w/2, -p.Ly/2), w, bot, facecolor=fc, edgecolor=ec, lw=0.8))
    # Glow azul nas aberturas
    for slit in [slit1, slit2]:
        sw = slit[1] - slit[0]
        for alpha, width in [(0.07, 1.2), (0.15, 0.6), (0.28, 0.3)]:
            ax.add_patch(mpatches.Rectangle(
                (xw - w*width/2, slit[0]), w*width, sw,
                facecolor='#00d4ff', alpha=alpha))
    # Rótulos F1 / F2
    for slit, lbl in [(slit1, 'F1'), (slit2, 'F2')]:
        ax.text(xw + p.Lx*0.016, (slit[0]+slit[1])/2, lbl,
                color='#00ff88', fontsize=6, va='center',
                fontfamily='monospace', alpha=0.85)


def _draw_detector(ax, xw, slit1, slit2, flash, t_val):
    col = '#ff3366' if flash else '#ff6688'
    pulse = 0.5 + 0.5 * np.sin(t_val * 8)
    dx_off = p.Lx * 0.022
    dw     = p.Lx * 0.030
    for slit in [slit1, slit2]:
        sc = (slit[0] + slit[1]) / 2
        sw = slit[1] - slit[0]
        dh = sw * 1.9
        # Caixa do detector
        ax.add_patch(mpatches.FancyBboxPatch(
            (xw + dx_off, sc - dh/2), dw, dh,
            boxstyle="round,pad=0.08",
            facecolor='#1a0010', edgecolor=col,
            lw=1.4 + flash*1.5, alpha=0.88))
        # LED piscante
        led_a = 0.95 if flash else (0.4 + 0.5*pulse)
        ax.plot(xw + dx_off + dw/2, sc, 'o',
                color=col, ms=4 + flash*2.5,
                alpha=led_a, markeredgecolor='#fff', markeredgewidth=0.3)
        # Feixe laser (medição)
        for i, a in enumerate([0.05, 0.12, 0.25]):
            beam_a = (a + 0.3*pulse*a) if not flash else a*2.5
            ax.plot([xw - p.Lx*0.012*(i+1), xw + dx_off],
                    [sc, sc], color=col, lw=0.5+i*0.3, alpha=beam_a)
        # Texto DET
        ax.text(xw + dx_off + dw/2, sc - dh/2 - p.Ly*0.03,
                'DET', color=col, fontsize=5.5, ha='center',
                fontfamily='monospace', alpha=0.8)
    # Onda de choque no colapso
    if flash:
        for r in [0.06, 0.14, 0.25]:
            theta = np.linspace(0, 2*np.pi, 100)
            cx = xw + p.Lx * 0.035
            rv = r * p.Ly
            ax.plot(cx + rv*np.cos(theta), rv*np.sin(theta),
                    color='#ff3366', lw=0.9, alpha=0.35*(1 - r*2.5))


def _draw_screen(ax, prob):
    sx = p.Lx * 0.96
    last = max(1, int(p.Nx * 0.06))
    profile = np.sum(np.abs(prob[-last:]), axis=0)
    if profile.max() > 0:
        profile /= profile.max()
    y_arr = np.linspace(-p.Ly/2, p.Ly/2, len(profile))
    ax.fill_betweenx(y_arr, sx, sx + profile * p.Lx * 0.028,
                     color='#ffffff', alpha=0.14)
    ax.plot(sx + profile * p.Lx * 0.028, y_arr,
            color='#ffffff', lw=0.8, alpha=0.5)
    ax.axvline(sx, color='#4a8aaa', lw=1.2, alpha=0.6)
    ax.text(sx + p.Lx*0.005, p.Ly/2 - p.Ly*0.06,
            'TELA', color='#4a8aaa', fontsize=5.5,
            fontfamily='monospace', alpha=0.75)


def render_profile_comparison(profA, profB, profD, y, det_prob=None):
    fig, ax = plt.subplots(figsize=(3.0, 5.8), facecolor='#020810')
    ax.set_facecolor('#020810')
    for sp in ax.spines.values():
        sp.set_edgecolor('#0a2040')
    ax.tick_params(colors='#3a6080', labelsize=7)

    if profA is not None:
        ax.fill_betweenx(y, 0, profA, color='#00d4ff', alpha=0.18)
        ax.plot(profA, y, color='#00d4ff', lw=1.5, label='A: Interferência')
    if profB is not None:
        ax.fill_betweenx(y, 0, profB, color='#ff7030', alpha=0.18)
        ax.plot(profB, y, color='#ff7030', lw=1.5, ls='--', label='B: Colapso VN')
    if profD is not None:
        ax.fill_betweenx(y, 0, profD, color='#00ffa0', alpha=0.12)
        ax.plot(profD, y, color='#00ffa0', lw=1.2, ls=':', label='D: Decoerência')

    # Teórico sinc²
    a_ = p.slit_width; L_ = p.Lx * (1 - p.x_wall_f)
    beta = np.where(np.abs(y) < 1e-9, 1e-9,
                    np.pi * a_ * y / (L_ * p.lam / (2*np.pi) + 1e-9))
    sc = np.where(np.abs(beta) < 1e-9, 1.0, (np.sin(beta)/beta)**2)
    if sc.max() > 0: sc /= sc.max()
    ax.plot(sc * 0.85, y, color='#ffcc00', lw=0.9, ls=':', alpha=0.6, label='sinc²(y)')

    ax.set_xlim(-0.05, 1.2)
    ax.set_xlabel('I(y)', color='#4a8aaa', fontsize=7)
    ax.set_ylabel('y (u.a.)', color='#4a8aaa', fontsize=7)
    ax.legend(fontsize=6, framealpha=0.3, facecolor='#020810',
              edgecolor='#0a2040', labelcolor='white', loc='lower right')
    ax.set_title('Tela Detectora', color='#6aaac0', fontsize=8, pad=2)
    if det_prob is not None:
        ax.text(0.05, -p.Ly/2 + p.Ly*0.05,
                f'P₁={det_prob:.3f}', color='#ff7030',
                fontsize=6.5, fontfamily='monospace')

    return fig_to_b64(fig, dpi=110)

# ══════════════════════════════════════════════════════════════════════════════
#  §4  SIMULAÇÃO — 4 cenários
# ══════════════════════════════════════════════════════════════════════════════

def run_scenario(label, x, y, X, Y, KX, KY, slit1, slit2, xw, iw,
                 mode="A", decoherence_strength=0.0, eraser_active=True):
    psi = gaussian_packet(X, Y)
    V, ab, _, _, _, _ = build_barrier(y)
    eT, eV, eAb = build_propagators(KX, KY, V, ab)

    Nt = p.Nt_pre + p.Nt_post
    fi = max(1, Nt // p.N_FRAMES)
    frames = []
    collapsed = False
    det_prob  = None
    dc_done   = False

    with Progress(
        SpinnerColumn(style="cyan"),
        BarColumn(bar_width=34, style="cyan"),
        TextColumn("[cyan]" + label + " {task.percentage:>3.0f}%"),
        TimeElapsedColumn(), console=console
    ) as prog:
        task = prog.add_task("", total=Nt)

        for step in range(Nt):

            # ── B: Colapso Von Neumann ────────────────────────────────────────
            if mode == "B" and step == p.Nt_pre and not collapsed:
                psi, det_prob = von_neumann_project(psi, X, Y, slit1, xw)
                collapsed = True
                console.print(f"   [bold red]⚡ Colapso Von Neumann — P₁={det_prob:.4f}[/bold red]")

            # ── D: Decoerência progressiva ────────────────────────────────────
            if mode == "D" and step > p.Nt_pre:
                t_since = step - p.Nt_pre
                gamma = decoherence_strength * (t_since / p.Nt_post) ** 0.7
                psi = apply_decoherence(psi, gamma)

            # ── Evolução Split-Operator ───────────────────────────────────────
            psi = so_step(psi, eT, eV, eAb)

            # ── DC: Escolha Atrasada (Wheeler) ────────────────────────────────
            if mode == "DC" and step == p.Nt_pre + p.Nt_post//2 and not dc_done:
                psi = delayed_choice_erase(psi, X, Y, xw, eraser=eraser_active)
                dc_done = True
                console.print(f"   [bold magenta]🌀 Escolha Atrasada — apagador={'ON' if eraser_active else 'OFF'}[/bold magenta]")

            # ── Renderiza frame ───────────────────────────────────────────────
            if step % fi == 0:
                prob = np.abs(psi)**2
                is_flash = (mode == "B" and collapsed and step <= p.Nt_pre + 4)
                t_val = step * p.dt
                cmap_key = mode if mode in CMAPS else "A"
                b64 = render_frame(
                    prob, iw, slit1, slit2, xw,
                    cmap_key=cmap_key,
                    detector_on=(mode == "B"),
                    detector_flash=is_flash,
                    t_val=t_val,
                    decoherence=decoherence_strength if mode == "D" else 0.0,
                )
                frames.append(b64)

            prog.update(task, advance=1)

    return frames, psi, det_prob

# ══════════════════════════════════════════════════════════════════════════════
#  §5  HTML — montagem com 3 abas (tabs funcionando corretamente)
# ══════════════════════════════════════════════════════════════════════════════

def build_html(frA, frB, frD, frDC, profB64, det_prob, Nt_pre_frac):
    fA_js  = json.dumps(frA)
    fB_js  = json.dumps(frB)
    fD_js  = json.dumps(frD)
    fDC_js = json.dumps(frDC)

    lam_s   = f"{p.lam:.3f}"
    k0_s    = str(p.k0)
    dlam_s  = str(p.slit_d)
    alam_s  = str(p.slit_a)
    v0_s    = str(int(p.V0))
    nx_s    = str(p.Nx)
    ny_s    = str(p.Ny)
    dt_s    = str(p.dt)
    nt_s    = str(p.Nt_pre + p.Nt_post)
    ekin_s  = f"{p.k0**2/2:.2f}"
    dp_s    = f"{det_prob:.4f}" if det_prob else "—"

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Dupla Fenda — Simulação Quântica TDSE 2D</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&family=Space+Grotesk:wght@300;400;600&display=swap');
*{{margin:0;padding:0;box-sizing:border-box}}
:root{{
  --bg:#010610;--panel:#040d1e;--border:#0b2240;--border2:#0a1830;
  --c:#00d4ff;--cm:#ff7030;--g:#00ff88;--w:#fff;--y:#ffcc00;--pu:#c060ff;
  --td:#2a5070;--dm:#030b1a;--fm:'Share Tech Mono',monospace;
}}
html,body{{height:100%;overflow:hidden;background:var(--bg);color:var(--c);font-family:var(--fm)}}
body::after{{content:'';position:fixed;inset:0;pointer-events:none;z-index:9999;
  background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,212,255,.010) 2px,rgba(0,212,255,.010) 4px)}}

/* ── LAYOUT ROOT ─────────────────────────────────────── */
/* Flex coluna: header fixo + tabs fixo + pane cresce     */
.root{{display:flex;flex-direction:column;height:100vh;overflow:hidden}}

/* ── HEADER ──────────────────────────────────────────── */
header{{
  flex:0 0 auto;
  display:flex;align-items:center;gap:16px;padding:0 22px;height:50px;
  background:linear-gradient(90deg,#05102a,#010610);
  border-bottom:1px solid var(--border);
  box-shadow:0 2px 24px rgba(0,212,255,.09);
}}
.logo{{font-family:'Orbitron',sans-serif;font-size:10px;font-weight:900;
  letter-spacing:5px;color:var(--c);text-shadow:0 0 20px var(--c)}}
.htitle{{font-family:'Orbitron',sans-serif;font-size:11px;font-weight:700;
  color:#fff;letter-spacing:2px}}
.hpill{{margin-left:auto;display:flex;align-items:center;gap:7px;
  font-size:8px;color:var(--g);letter-spacing:3px}}
.dot{{width:7px;height:7px;border-radius:50%;background:var(--g);
  box-shadow:0 0 8px var(--g);animation:blink 1.4s ease-in-out infinite}}
@keyframes blink{{0%,100%{{opacity:1}}50%{{opacity:.25}}}}

/* ── TABS ────────────────────────────────────────────── */
.tabs{{
  flex:0 0 auto;
  display:flex;align-items:stretch;height:36px;
  background:var(--dm);border-bottom:1px solid var(--border);padding:0 12px;
}}
.tab{{
  padding:0 20px;font-size:9px;letter-spacing:2px;text-transform:uppercase;
  cursor:pointer;border-bottom:2px solid transparent;
  display:flex;align-items:center;gap:6px;
  color:var(--td);transition:all .2s;white-space:nowrap;user-select:none;
}}
.tab:hover{{color:var(--w)}}
.tab.active{{color:var(--c);border-bottom-color:var(--c)}}

/* ── PANES ───────────────────────────────────────────── */
/* All panes hidden by default.                          */
/* JS sets display correctly per pane type.              */
.pane{{display:none;flex:1 1 auto;overflow:hidden;min-height:0}}

/* ── PANE SIMULAÇÃO ──────────────────────────────────── */
#paneSimu{{display:none}}          /* JS sets to grid when active */

.sidebar,.rpanel{{
  overflow-y:auto;background:var(--panel);
}}
.sidebar{{
  border-right:1px solid var(--border);
  padding:14px 12px;
  display:flex;flex-direction:column;gap:12px;
}}
.rpanel{{
  border-left:1px solid var(--border);
  padding:14px 12px;
  display:flex;flex-direction:column;gap:10px;
}}
.slabel{{font-size:7.5px;letter-spacing:3px;color:var(--td);text-transform:uppercase;
  border-bottom:1px solid var(--border);padding-bottom:4px;margin-bottom:4px}}
.ctrl-row{{display:flex;flex-direction:column;gap:3px;margin-bottom:6px}}
.ctrl-row label{{display:flex;justify-content:space-between;font-size:8.5px}}
.ctrl-row label span{{color:var(--c)}}
input[type=range]{{-webkit-appearance:none;width:100%;height:2px;
  background:var(--border);border-radius:2px;outline:none}}
input[type=range]::-webkit-slider-thumb{{-webkit-appearance:none;width:12px;height:12px;
  border-radius:50%;background:var(--c);box-shadow:0 0 8px var(--c);cursor:pointer}}
.tog{{display:flex;align-items:center;justify-content:space-between;
  padding:6px 8px;background:rgba(0,212,255,.04);
  border:1px solid var(--border);border-radius:3px;
  cursor:pointer;user-select:none;font-size:8.5px;transition:all .2s}}
.tog:hover{{border-color:var(--c)}}
.tog.on{{border-color:var(--cm);background:rgba(255,112,48,.08);color:var(--cm)}}
.sw{{width:28px;height:14px;background:var(--border);border-radius:7px;
  position:relative;transition:background .2s}}
.tog.on .sw{{background:var(--cm)}}
.sw::after{{content:'';position:absolute;width:10px;height:10px;border-radius:50%;
  background:#fff;top:2px;left:2px;transition:transform .2s}}
.tog.on .sw::after{{transform:translateX(14px)}}
.radio-g{{display:flex;gap:4px}}
.radio-g label{{flex:1;padding:5px 3px;border:1px solid var(--border);
  border-radius:2px;text-align:center;cursor:pointer;font-size:7.5px;
  letter-spacing:1px;transition:all .2s;user-select:none}}
.radio-g label:hover{{border-color:var(--c);color:var(--c)}}
.radio-g input{{display:none}}
.radio-g input:checked + label{{border-color:var(--c);color:var(--c);
  background:rgba(0,212,255,.10);box-shadow:0 0 8px rgba(0,212,255,.15)}}
.btn{{width:100%;padding:8px 4px;border:1px solid var(--c);background:transparent;
  color:var(--c);font-family:var(--fm);font-size:8.5px;letter-spacing:2px;
  cursor:pointer;text-transform:uppercase;border-radius:2px;transition:all .2s}}
.btn:hover{{background:rgba(0,212,255,.10);box-shadow:0 0 14px rgba(0,212,255,.2)}}
.btn.red{{border-color:#ff4466;color:#ff4466}}
.btn.red:hover{{background:rgba(255,68,102,.10)}}
.btn.sm{{padding:5px 4px;font-size:7.5px;letter-spacing:1.5px}}
.info{{padding:8px;border:1px solid var(--border);border-radius:3px;
  font-size:8px;line-height:1.75;color:var(--td);background:rgba(0,212,255,.025)}}
.info b{{color:var(--c)}}
.info.om b{{color:var(--cm)}}
.info.gn b{{color:var(--g)}}
.info.pu b{{color:var(--pu)}}
.eq{{padding:8px 10px;border:1px solid var(--border);border-radius:3px;
  font-size:8.5px;line-height:2.1;color:var(--td);background:rgba(0,0,0,.5)}}
.eq .ec{{color:var(--c)}} .eq .em{{color:var(--cm)}} .eq .eg{{color:var(--g)}}
.stat{{background:rgba(0,212,255,.03);border:1px solid var(--border);
  border-radius:2px;padding:7px 8px;margin-bottom:5px}}
.stat .sn{{font-size:7.5px;color:var(--td);letter-spacing:2px;margin-bottom:2px}}
.stat .sv{{font-family:'Orbitron',sans-serif;font-size:14px;color:var(--c);font-weight:700}}
.stat .su{{font-size:7px;color:var(--td);margin-left:2px}}
.legend-item{{display:flex;align-items:center;gap:7px;font-size:8px;color:var(--td);line-height:2.2}}
.ll{{width:18px;height:2px;border-radius:1px;flex-shrink:0}}
#profileImg{{width:100%;border:1px solid var(--border);border-radius:2px;display:block}}

/* Main canvas */
.main{{position:relative;display:flex;flex-direction:column;
  background:radial-gradient(ellipse at 48% 50%,#010c1e,#010408 72%);overflow:hidden}}
.canvas-wrap{{position:relative;flex:1;display:flex;align-items:center;
  justify-content:center;overflow:hidden}}
#simImg{{max-width:100%;max-height:100%;display:block}}
.ovl{{position:absolute;pointer-events:none;font-size:7.5px;letter-spacing:2px;color:var(--td)}}
.ovl.tl{{top:9px;left:11px}} .ovl.tr{{top:9px;right:11px}} .ovl.bl{{bottom:6px;left:11px}}
.badge{{padding:3px 10px;border:1px solid;border-radius:2px;font-size:7.5px;letter-spacing:3px;text-transform:uppercase}}
.badge.A{{border-color:var(--c);color:var(--c);background:rgba(0,212,255,.07)}}
.badge.B{{border-color:var(--cm);color:var(--cm);background:rgba(255,112,48,.07)}}
.badge.D{{border-color:var(--g);color:var(--g);background:rgba(0,255,136,.05)}}
.badge.DC{{border-color:var(--pu);color:var(--pu);background:rgba(192,96,255,.06)}}
.pbar-wrap{{height:2px;flex:0 0 auto;width:100%;background:var(--border)}}
.pbar{{height:100%;width:0%;transition:width .08s}}
.pbar.A{{background:var(--c);box-shadow:0 0 5px var(--c)}}
.pbar.B{{background:var(--cm);box-shadow:0 0 5px var(--cm)}}
.pbar.D{{background:var(--g);box-shadow:0 0 5px var(--g)}}
.pbar.DC{{background:var(--pu);box-shadow:0 0 5px var(--pu)}}
.strip{{display:flex;align-items:center;gap:14px;padding:5px 12px;flex:0 0 auto;
  border-top:1px solid var(--border);background:var(--dm)}}
.sv{{font-size:8px;display:flex;gap:4px}}
.sv .k{{color:var(--td)}} .sv .v{{color:var(--w)}} .sv .vc{{color:var(--c)}}
.sv .vm{{color:var(--cm)}} .sv .vy{{color:var(--y)}}
@keyframes cflash{{0%{{opacity:0}}15%{{opacity:1}}100%{{opacity:0}}}}
.cflash{{position:absolute;inset:0;pointer-events:none;opacity:0;
  background:radial-gradient(ellipse at 40% 50%,rgba(255,51,100,.20),transparent 65%)}}
.cflash.go{{animation:cflash 1.4s ease-out forwards}}
@keyframes fadein{{from{{opacity:0;transform:translateY(-4px)}}to{{opacity:1;transform:none}}}}
.alert{{display:none;align-items:center;gap:7px;padding:5px 9px;
  border:1px solid #ff3366;border-radius:3px;background:rgba(255,51,102,.1);
  font-size:8px;color:#ff6688;letter-spacing:1px;animation:fadein .3s}}
.alert.show{{display:flex}}

/* ── PANE FÍSICA ─────────────────────────────────────── */
#paneFisica{{overflow-y:auto}}     /* JS sets to block when active */
#paneFisica.inner{{padding:32px 48px;background:var(--bg)}}
.ph1{{font-family:'Orbitron',sans-serif;font-size:18px;color:var(--c);
  letter-spacing:3px;margin-bottom:6px}}
.psub{{color:var(--td);font-size:10px;letter-spacing:2px;margin-bottom:32px}}
.pgrid{{display:grid;grid-template-columns:1fr 1fr;gap:22px;max-width:1100px}}
.pcard{{background:var(--panel);border:1px solid var(--border);border-radius:6px;
  padding:20px;display:flex;flex-direction:column;gap:12px}}
.pcard h2{{font-family:'Orbitron',sans-serif;font-size:10.5px;letter-spacing:3px;
  border-bottom:1px solid var(--border);padding-bottom:8px}}
.pcard h2.cc{{color:var(--c)}} .pcard h2.cm{{color:var(--cm)}}
.pcard h2.cg{{color:var(--g)}} .pcard h2.cp{{color:var(--pu)}} .pcard h2.cy{{color:var(--y)}}
.pt{{font-family:'Space Grotesk',sans-serif;font-size:13px;line-height:1.8;
  color:#8ab4c8;font-weight:300}}
.pt b{{color:var(--w);font-weight:600}}
.pt .hc{{color:var(--c)}} .pt .hm{{color:var(--cm)}}
.pt .hg{{color:var(--g)}} .pt .hp{{color:var(--pu)}} .pt .hy{{color:var(--y)}}
.eqb{{background:rgba(0,0,0,.5);border:1px solid var(--border);border-radius:4px;
  padding:13px 16px;font-family:var(--fm);font-size:11px;line-height:2.2;color:var(--td)}}
.eqb .me{{color:var(--c);font-size:13px}}
.eqb .se{{color:#4a7090;font-size:9px;margin-left:10px}}
.eqb .he{{color:var(--y)}}
.psep{{grid-column:1/-1;border:none;border-top:1px solid var(--border);margin:2px 0}}

/* ── PANE INTERPRETAÇÕES ─────────────────────────────── */
#paneInterp{{overflow-y:auto}}     /* JS sets to block when active */
#paneInterp.inner{{padding:28px 36px;background:var(--bg)}}
.ip1{{font-family:'Orbitron',sans-serif;font-size:16px;color:var(--pu);
  letter-spacing:3px;margin-bottom:6px}}
.ipsub{{color:var(--td);font-size:10px;margin-bottom:26px}}
.igrid{{display:grid;grid-template-columns:1fr 1fr;gap:18px;max-width:1100px}}
.icard{{background:var(--panel);border:1px solid var(--border);border-radius:6px;padding:20px}}
.icard.co{{border-color:rgba(0,212,255,.3)}}
.icard.mw{{border-color:rgba(192,96,255,.3)}}
.icard.pi{{border-color:rgba(0,255,136,.25)}}
.icard.cs{{border-color:rgba(255,204,0,.2)}}
.icard h2{{font-family:'Orbitron',sans-serif;font-size:10.5px;letter-spacing:3px;
  padding-bottom:8px;margin-bottom:12px;border-bottom:1px solid var(--border)}}
.icard.co h2{{color:var(--c)}} .icard.mw h2{{color:var(--pu)}}
.icard.pi h2{{color:var(--g)}} .icard.cs h2{{color:var(--y)}}
.ilbl{{font-size:8px;letter-spacing:2px;color:var(--td);text-transform:uppercase;margin-bottom:4px}}
.it{{font-family:'Space Grotesk',sans-serif;font-size:12.5px;line-height:1.75;
  color:#8ab4c8;font-weight:300;margin-bottom:10px}}
.it b{{color:var(--w);font-weight:600}}
.vb{{padding:10px 12px;border-radius:4px;font-size:9.5px;font-family:var(--fm);
  letter-spacing:1px;line-height:1.8;margin-top:4px}}
.vb.A{{border:1px solid rgba(0,212,255,.3);background:rgba(0,212,255,.05);color:var(--c)}}
.vb.B{{border:1px solid rgba(192,96,255,.3);background:rgba(192,96,255,.05);color:var(--pu)}}
.vb.C{{border:1px solid rgba(0,255,136,.25);background:rgba(0,255,136,.04);color:var(--g)}}
.vb.D{{border:1px solid rgba(255,204,0,.2);background:rgba(255,204,0,.04);color:var(--y)}}
.vsdiv{{grid-column:1/-1;display:flex;align-items:center;gap:14px;margin:2px 0}}
.vsdiv::before,.vsdiv::after{{content:'';flex:1;height:1px;background:var(--border)}}
.vsdiv span{{font-family:'Orbitron',sans-serif;font-size:9px;color:var(--td);letter-spacing:3px}}
.ctbl{{width:100%;border-collapse:collapse;font-size:9.5px;font-family:var(--fm)}}
.ctbl th{{text-align:left;padding:7px 10px;letter-spacing:2px;color:var(--td);
  border-bottom:1px solid var(--border)}}
.ctbl td{{padding:6px 10px;border-bottom:1px solid var(--border2);color:#6a9aaa}}

::-webkit-scrollbar{{width:3px}}
::-webkit-scrollbar-track{{background:var(--bg)}}
::-webkit-scrollbar-thumb{{background:var(--border);border-radius:2px}}
</style>
</head>
<body>
<div class="root">

<!-- HEADER -->
<header>
  <div class="logo">QM//TDSE</div>
  <div class="htitle">EXPERIMENTO DA DUPLA FENDA — Simulação Quântica 2D</div>
  <div id="badgeWrap" style="margin-left:20px"><span class="badge A">MODO ONDA</span></div>
  <div class="hpill"><div class="dot"></div>SISTEMA ATIVO</div>
</header>

<!-- TABS -->
<div class="tabs">
  <div class="tab active" id="tabSimu"   onclick="switchTab('Simu')">⚛ SIMULAÇÃO</div>
  <div class="tab"        id="tabFisica" onclick="switchTab('Fisica')">📐 FÍSICA &amp; EQUAÇÕES</div>
  <div class="tab"        id="tabInterp" onclick="switchTab('Interp')">🌌 INTERPRETAÇÕES</div>
</div>

<!-- ══════════════════════════════════════
     PANE SIMULAÇÃO
══════════════════════════════════════ -->
<div class="pane" id="paneSimu">
  <!-- Sidebar -->
  <div class="sidebar">
    <div>
      <div class="slabel">Cenário</div>
      <div class="radio-g" style="margin-bottom:5px">
        <input type="radio" name="scen" id="rA"  value="A"  checked><label for="rA">A: ONDA</label>
        <input type="radio" name="scen" id="rB"  value="B"><label for="rB">B: COLAPSO</label>
      </div>
      <div class="radio-g">
        <input type="radio" name="scen" id="rD"  value="D"><label for="rD">D: DECOER.</label>
        <input type="radio" name="scen" id="rDC" value="DC"><label for="rDC">DC: ATRAS.</label>
      </div>
    </div>

    <div>
      <div class="slabel">Modo Emissão</div>
      <div class="radio-g">
        <input type="radio" name="emit" id="emM" value="multi"  checked><label for="emM">MÚLTIPLAS</label>
        <input type="radio" name="emit" id="emS" value="single"><label for="emS">INDIVIDUAL</label>
      </div>
    </div>

    <div>
      <div class="slabel">Observação</div>
      <div class="tog" id="togDet" style="margin-bottom:5px">
        <span>Detector Quais-Fenda</span><div class="sw"></div>
      </div>
      <div class="tog" id="togDelay">
        <span>Escolha Atrasada</span><div class="sw"></div>
      </div>
    </div>

    <div>
      <div class="slabel">Controles</div>
      <div class="ctrl-row">
        <label>Velocidade <span id="spV">2×</span></label>
        <input type="range" id="spR" min="1" max="6" value="2">
      </div>
      <button class="btn"    id="btnPlay"  style="margin-bottom:4px">▶ PLAY</button>
      <button class="btn sm" id="btnReset" style="margin-bottom:4px">↺ REINICIAR</button>
      <button class="btn sm red" id="btnStop">■ PARAR</button>
    </div>

    <div class="eq">
      <div><span class="ec">iℏ∂Ψ/∂t</span> = ĤΨ</div>
      <div><span class="ec">Û(dt)</span> ≈ e⁻ⁱᵛᵈᵗ/²·T̂·e⁻ⁱᵛᵈᵗ/²</div>
      <div><span class="em">P̂₁|Ψ⟩</span> → colapso F1</div>
      <div><span class="eg">Γ·Ψ</span> → decoerência</div>
      <div style="margin-top:4px;font-size:7.5px">λ={lam_s} | k₀={k0_s}</div>
      <div style="font-size:7.5px">d/λ={dlam_s} | a/λ={alam_s}</div>
    </div>

    <div class="info" id="scenInfo">
      <b>Cenário A — Interferência</b><br>
      Sem observação. Ψ=Ψ₁+Ψ₂. Padrão de franjas cos²·sinc².
    </div>

    <div class="alert" id="colAlert">⚡ COLAPSO Von Neumann aplicado</div>
  </div>

  <!-- Main canvas -->
  <div class="main">
    <div class="canvas-wrap">
      <div class="ovl tl" id="ovlTL">t = 0.000 u.a.</div>
      <div class="ovl bl">← FONTE  ·  BARREIRA  ·  TELA DETECTORA →</div>
      <div class="ovl tr" id="badgeEl"><span class="badge A">MODO ONDA</span></div>
      <img id="simImg" src="" alt="Simulação TDSE">
      <div class="cflash" id="cflash"></div>
    </div>
    <div class="pbar-wrap"><div class="pbar A" id="pbar"></div></div>
    <div class="strip">
      <div class="sv"><span class="k">FRAME</span><span class="v"  id="sfn">—</span></div>
      <div class="sv"><span class="k">t</span>    <span class="vc" id="st">—</span></div>
      <div class="sv"><span class="k">||Ψ||²</span><span class="vc" id="snorm">—</span></div>
      <div class="sv"><span class="k">P₁</span>  <span class="vy" id="sp1">—</span></div>
      <div class="sv"><span class="k">MODO</span> <span class="vc" id="smodo">ONDA</span></div>
      <div class="sv"><span class="k">EMISSÃO</span><span class="v" id="semit">MÚLTIPLAS</span></div>
      <div style="margin-left:auto" class="sv">
        <span class="k">Split-Operator | TDSE 2D | {nx_s}×{ny_s} | dt={dt_s} | {nt_s} steps</span>
      </div>
    </div>
  </div>

  <!-- Right panel -->
  <div class="rpanel">
    <div>
      <div class="slabel">Estatísticas</div>
      <div class="stat"><div class="sn">λ DE BROGLIE</div><div class="sv">{lam_s}<span class="su">u.a.</span></div></div>
      <div class="stat"><div class="sn">d / λ</div><div class="sv">{dlam_s}</div></div>
      <div class="stat"><div class="sn">a / λ</div><div class="sv">{alam_s}</div></div>
      <div class="stat"><div class="sn">E CINÉTICA</div><div class="sv">{ekin_s}<span class="su">u.a.</span></div></div>
      <div class="stat"><div class="sn">PROB. DETECÇÃO P₁</div><div class="sv" id="p1stat">—</div></div>
    </div>
    <div>
      <div class="slabel">Perfil I(y) — Tela</div>
      <img id="profileImg" src="data:image/png;base64,{profB64}" alt="Perfil">
    </div>
    <div>
      <div class="slabel">Legenda</div>
      <div class="legend-item"><div class="ll" style="background:#00d4ff"></div>A: interferência</div>
      <div class="legend-item"><div class="ll" style="background:#ff7030"></div>B: colapso VN</div>
      <div class="legend-item"><div class="ll" style="background:#00ffa0"></div>D: decoerência</div>
      <div class="legend-item"><div class="ll" style="background:#00ff88;opacity:.7"></div>barreira / fendas</div>
      <div class="legend-item"><div class="ll" style="background:#4a8aaa"></div>anteparo final</div>
      <div class="legend-item"><div class="ll" style="background:#ff3366"></div>detector ativo</div>
    </div>
    <div class="info" style="font-size:7.5px">
      <b>Parâmetros:</b><br>
      V₀={v0_s} u.a. | k₀={k0_s}<br>
      E_cin={ekin_s} u.a.<br>
      Grade: {nx_s}×{ny_s}<br>
      Split-Operator Fourier<br>
      Von Neumann P={dp_s}<br>
      PML nas bordas<br><br>
      <b>Teclas:</b><br>
      ESPAÇO: play/pause<br>
      ← →: frame a frame<br>
      R: reiniciar<br>
      1/2/3/4: cenário
    </div>
  </div>
</div><!-- #paneSimu -->


<!-- ══════════════════════════════════════
     PANE FÍSICA
══════════════════════════════════════ -->
<div class="pane" id="paneFisica">
<div class="inner" style="padding:32px 48px;background:var(--bg);min-height:100%">
<h1 class="ph1">FÍSICA DO EXPERIMENTO</h1>
<p class="psub">Fundamentos quânticos, equações e método numérico Split-Operator</p>
<div class="pgrid">

<div class="pcard">
  <h2 class="cc">① EQUAÇÃO DE SCHRÖDINGER (TDSE)</h2>
  <p class="pt">O comportamento quântico é governado pela <b>TDSE</b>.
    A <span class="hc">função de onda Ψ(x,y,t)</span> contém toda a informação do sistema.
    A <b>densidade de probabilidade</b> de encontrar a partícula em (x,y) é <span class="hc">|Ψ|²</span>.</p>
  <div class="eqb">
    <div class="me">iℏ ∂Ψ/∂t = ĤΨ = (T̂ + V̂)Ψ</div>
    <div class="se">T̂ = −ℏ²/2m ∇²  (operador cinético)</div>
    <div class="se">V̂ = V(x,y)       (potencial da barreira)</div>
    <div class="se he">ℏ = m = 1  (unidades atômicas reduzidas)</div>
  </div>
  <p class="pt">T̂ é <b>diagonal no espaço k</b>: T̂(k) = k²/2. Isso motiva o <b>método Split-Operator</b>.</p>
</div>

<div class="pcard">
  <h2 class="cc">② PACOTE DE ONDAS GAUSSIANO</h2>
  <p class="pt">Estado inicial: <b>Gaussiano 2D</b> com momento ⟨p̂x⟩ = ℏk₀.
    O envelope Gaussiano localiza a partícula; exp(ik₀x) dá a propagação em X.</p>
  <div class="eqb">
    <div class="me">Ψ(x,y,0) = A · G(x,σₓ) · G(y,σᵧ) · e^(ik₀x)</div>
    <div class="se">G(ξ,σ) = exp[−(ξ−ξ₀)²/2σ²]</div>
    <div class="se">k₀ = {k0_s} → λ = {lam_s} u.a.</div>
    <div class="se">E_cin = k₀²/2 = {ekin_s} u.a.</div>
    <div class="se he">A normaliza: ∫∫|Ψ|² dx dy = 1</div>
  </div>
  <p class="pt">A <b>dispersão natural</b> do pacote — σ(t) cresce com t — emerge automaticamente da TDSE.</p>
</div>

<div class="pcard">
  <h2 class="cm">③ POTENCIAL DE BARREIRA V(x,y)</h2>
  <p class="pt"><b>V₀ ≫ E_cin</b> em toda a parede, exceto nas <b>duas aberturas</b> (V=0).
    A barreira força o pacote a difratat pelas fendas.</p>
  <div class="eqb">
    <div class="me">V(x,y) = V₀ · [1 − θ(F₁) − θ(F₂)] · δ_wall(x)</div>
    <div class="se">V₀ = {v0_s} u.a. ≫ E_cin = {ekin_s} u.a.</div>
    <div class="se">d/λ = {dlam_s}  |  a/λ = {alam_s}</div>
    <div class="se he">Franjas: d·sin(θ_m) = mλ  (m = 0, ±1, ±2, ...)</div>
  </div>
  <p class="pt">As bordas têm <b>PML (Perfectly Matched Layer)</b> para absorver reflexões da FFT.</p>
</div>

<div class="pcard">
  <h2 class="cg">④ SPLIT-OPERATOR FOURIER</h2>
  <p class="pt">Fatoração Trotter-Suzuki do propagador unitário. Operadores diagonais em domínios complementares.</p>
  <div class="eqb">
    <div class="me">Û(dt) ≈ e^(−iVdt/2) · e^(−iTdt) · e^(−iVdt/2)</div>
    <div class="se">Erro: O(dt³) por passo</div>
    <div class="se">1: Ψ ← exp(−iV·dt/2)·Ψ       [espaço real]</div>
    <div class="se">2: Ψ̃ ← FFT[Ψ]</div>
    <div class="se">3: Ψ̃ ← exp(−ik²dt/2)·Ψ̃       [espaço k]</div>
    <div class="se">4: Ψ ← iFFT[Ψ̃]</div>
    <div class="se">5: Ψ ← exp(−iV·dt/2)·Ψ</div>
    <div class="se he">O(N²logN) por passo | dt={dt_s} | Grade {nx_s}×{ny_s}</div>
  </div>
</div>

<hr class="psep">

<div class="pcard">
  <h2 class="cm">⑤ PROJEÇÃO DE VON NEUMANN</h2>
  <p class="pt">Ao medir "qual fenda", o estado colapsa: <b>Regra de Born</b>.
    A coerência entre Ψ₁ e Ψ₂ é <span class="hm">irreversivelmente destruída</span>.</p>
  <div class="eqb">
    <div class="me">Antes:  Ψ = α|ψ₁⟩ + β|ψ₂⟩   (superposição)</div>
    <div class="me">Depois: Ψ_new = P̂₁|Ψ⟩ / ||P̂₁|Ψ⟩||</div>
    <div class="se">P̂₁ = ∬(F1) |x,y⟩⟨x,y| dx dy</div>
    <div class="se">P₁ = ⟨Ψ|P̂₁|Ψ⟩ = ∬(F1)|Ψ|² dxdy = {dp_s}</div>
    <div class="se he">Resultado: padrão sinc²(y) — fenda única</div>
  </div>
</div>

<div class="pcard">
  <h2 class="cp">⑥ DECOERÊNCIA QUÂNTICA</h2>
  <p class="pt"><b>Decoerência</b>: acoplamento com o ambiente suprime off-diagonais da matriz densidade
    sem medição explícita. Explica a transição quântica→clássica.</p>
  <div class="eqb">
    <div class="me">ρ_ij(t) = ρ_ij(0) · exp(−Γ_ij · t)</div>
    <div class="se">Γ_ij: taxa de decoerência (acoplamento ambiental)</div>
    <div class="se">Γ→0: interferência plena</div>
    <div class="se he">Γ→∞: mistura clássica (dois picos gaussianos)</div>
  </div>
</div>

<div class="pcard">
  <h2 class="cp">⑦ ESCOLHA ATRASADA (WHEELER)</h2>
  <p class="pt">A decisão de medir é tomada <em>após</em> a partícula passar. O resultado retroativamente muda o padrão.</p>
  <div class="eqb">
    <div class="me">t₁: partícula passa pela fenda</div>
    <div class="me">t₂ &gt; t₁: decisão de apagar ou não</div>
    <div class="se">Apagador ON  → restaura interferência</div>
    <div class="se">Apagador OFF → mantém padrão fenda única</div>
    <div class="se he">Não viola causalidade — correlações pós-selecionadas</div>
  </div>
</div>

<div class="pcard">
  <h2 class="cy">⑧ PADRÃO DE FRANJAS</h2>
  <p class="pt">Intensidade na tela: envelope de <b>difração</b> (sinc²) modulado pelas <b>franjas de interferência</b> (cos²).</p>
  <div class="eqb">
    <div class="me">I(y) = I₀ · sinc²(πa·sinθ/λ) · cos²(πd·sinθ/λ)</div>
    <div class="se">sinc²: difração de fenda única (largura a)</div>
    <div class="se">cos²:  interferência de dupla fenda (separação d)</div>
    <div class="se">Máximos: y_m = mλL/d   (m = 0, ±1, ±2, ...)</div>
    <div class="se he">d/λ={dlam_s} → franjas separadas por L/{dlam_s}</div>
  </div>
</div>

</div><!-- pgrid -->
</div><!-- inner -->
</div><!-- #paneFisica -->


<!-- ══════════════════════════════════════
     PANE INTERPRETAÇÕES
══════════════════════════════════════ -->
<div class="pane" id="paneInterp">
<div class="inner" style="padding:28px 36px;background:var(--bg);min-height:100%">
<h1 class="ip1">INTERPRETAÇÕES DA MECÂNICA QUÂNTICA</h1>
<p class="ipsub">O que realmente acontece quando a partícula "passa pelas fendas"?</p>
<div class="igrid">

<div class="icard co">
  <h2>① COPENHAGUE</h2>
  <div class="ilbl">Proponentes</div>
  <p class="it">Bohr, Heisenberg (1927). Interpretação dominante na física aplicada.</p>
  <div class="ilbl">Sobre a dupla fenda</div>
  <p class="it">Antes da medição, a partícula <b>não tem posição definida</b>. Ψ é apenas um instrumento de cálculo.
    A partícula passa "por ambas" no sentido matemático. Ao medir, Ψ <b>colapsa instantaneamente</b>.
    A interferência desaparece porque a medição pertuba irreversivelmente o sistema.</p>
  <div class="ilbl">Realidade</div>
  <p class="it">Só existe quando observada. Há um corte observador-sistema indefinido.</p>
  <div class="vb A">✓ Colapso instantâneo ao medir<br>✓ Sem interferência quando detectado<br>✗ Mecanismo do colapso não explicado<br>✗ "Corte de Heisenberg" é indefinido</div>
</div>

<div class="icard mw">
  <h2>② MUITOS MUNDOS (MWI)</h2>
  <div class="ilbl">Proponentes</div>
  <p class="it">Hugh Everett III (1957). DeWitt, Deutsch, Carroll, Tegmark.</p>
  <div class="ilbl">Sobre a dupla fenda</div>
  <p class="it">Ψ <b>nunca colapsa</b>. A partícula passa pelas duas fendas em <b>dois ramos da realidade</b>.
    Ao medir, o universo se <b>divide (branches)</b> — o observador se entangleia com a partícula.
    A interferência some porque os ramos se tornam ortogonais (decoeridos).</p>
  <div class="ilbl">Realidade</div>
  <p class="it">Todos os resultados ocorrem em ramos paralelos do universo.</p>
  <div class="vb B">✓ Ψ sempre unitária — sem colapso ad hoc<br>✓ Decoerência emerge naturalmente<br>✗ Proliferação exponencial de ramos<br>✗ Probabilidades Born não trivialmente derivadas</div>
</div>

<div class="vsdiv"><span>OUTRAS INTERPRETAÇÕES</span></div>

<div class="icard pi">
  <h2>③ ONDA PILOTO (BOHM)</h2>
  <div class="ilbl">Proponentes</div>
  <p class="it">de Broglie (1927), Bohm (1952). Variáveis ocultas não-locais.</p>
  <div class="ilbl">Sobre a dupla fenda</div>
  <p class="it">A partícula tem <b>posição real e definida</b> sempre, guiada por uma "onda piloto" que satisfaz Schrödinger.
    A onda passa por <em>ambas</em>; a partícula por <em>uma só</em>, mas sua trajetória é determinada pelo <b>potencial quântico Q</b>.</p>
  <div class="ilbl">Realidade</div>
  <p class="it">Determinística, realista, mas <b>não-local</b>.</p>
  <div class="vb C">✓ Trajetórias determinísticas bem definidas<br>✓ Medição sem colapso misterioso<br>✗ Não-local (coerente com Bell)<br>✗ Potencial quântico sem análogo clássico</div>
</div>

<div class="icard cs">
  <h2>④ HISTÓRIAS CONSISTENTES</h2>
  <div class="ilbl">Proponentes</div>
  <p class="it">Griffiths, Omnès, Gell-Mann &amp; Hartle (1980s).</p>
  <div class="ilbl">Sobre a dupla fenda</div>
  <p class="it">"A partícula foi pela fenda 1" e "pela fenda 2" são histórias <em>inconsistentes</em> quando há interferência —
    <b>não se pode afirmar ambas simultaneamente</b>. Com detector, tornam-se consistentes e a interferência some.</p>
  <div class="ilbl">Realidade</div>
  <p class="it">Descrita por frameworks de histórias mutuamente exclusivos.</p>
  <div class="vb D">✓ Matematicamente rigorosa<br>✓ Elimina o "observador" como elemento fundamental<br>✗ Múltiplos frameworks igualmente válidos<br>✗ Pouco intuitiva</div>
</div>

<div class="vsdiv"><span>COMPARATIVO DIRETO</span></div>

<div class="icard" style="grid-column:1/-1;border-color:rgba(255,204,0,.2)">
  <h2 style="color:var(--y)">⑤ COMPARATIVO: A DUPLA FENDA PELAS 4 INTERPRETAÇÕES</h2>
  <div style="overflow-x:auto">
  <table class="ctbl">
    <thead><tr>
      <th>QUESTÃO</th>
      <th style="color:var(--c)">COPENHAGUE</th>
      <th style="color:var(--pu)">MUITOS MUNDOS</th>
      <th style="color:var(--g)">ONDA PILOTO</th>
      <th style="color:var(--y)">HIST. CONSIST.</th>
    </tr></thead>
    <tbody>
      <tr><td>Por quantas fendas passa?</td><td style="color:var(--c)">Ambas (onda)</td><td style="color:var(--pu)">Ambas (2 ramos)</td><td style="color:var(--g)">Uma só (guiada)</td><td style="color:var(--y)">Pergunta inválida</td></tr>
      <tr><td>O colapso é real?</td><td style="color:var(--c)">Sim, físico</td><td style="color:var(--pu)">Não, aparente</td><td style="color:var(--g)">Não (guiagem)</td><td style="color:var(--y)">Mudança framework</td></tr>
      <tr><td>O observador importa?</td><td style="color:var(--c)">Sim, fundamental</td><td style="color:var(--pu)">Entra no estado</td><td style="color:var(--g)">Não fundamental</td><td style="color:var(--y)">Define framework</td></tr>
      <tr><td>Posição antes de medir?</td><td style="color:var(--c)">Não existe</td><td style="color:var(--pu)">Todos os ramos</td><td style="color:var(--g)">Existe e definida</td><td style="color:var(--y)">Sem sentido único</td></tr>
      <tr><td>Aceita por físicos?</td><td style="color:var(--c)">~60% (majoritária)</td><td style="color:var(--pu)">~20% (crescendo)</td><td style="color:var(--g)">~5% (nicho)</td><td style="color:var(--y)">~10%</td></tr>
    </tbody>
  </table>
  </div>
  <p class="it" style="margin-top:12px">
    <b>Nota:</b> todas as interpretações produzem as <em>mesmas previsões experimentais</em>.
    A escolha entre elas é <b style="color:var(--y)">filosófica, não empírica</b>.
  </p>
</div>

</div><!-- igrid -->
</div><!-- inner -->
</div><!-- #paneInterp -->

</div><!-- .root -->

<script>
// ── DADOS ────────────────────────────────────────────────────────
const FRAMES  = {{ A:{fA_js}, B:{fB_js}, D:{fD_js}, DC:{fDC_js} }};
const DET_PROB   = {det_prob if det_prob else 0.0};
const PRE_FRAC   = {Nt_pre_frac:.4f};
const DT         = {p.dt};
const NT_TOTAL   = {p.Nt_pre + p.Nt_post};

// ── STATE ────────────────────────────────────────────────────────
let scenario  = 'A';
let curFrame  = 0;
let playing   = false;
let timer     = null;
let speed     = 2;
let collapsed = false;

// ── DOM ──────────────────────────────────────────────────────────
const simImg   = document.getElementById('simImg');
const pbar     = document.getElementById('pbar');
const ovlTL    = document.getElementById('ovlTL');
const sfn      = document.getElementById('sfn');
const st_el    = document.getElementById('st');
const snorm    = document.getElementById('snorm');
const sp1_el   = document.getElementById('sp1');
const smodo    = document.getElementById('smodo');
const semit    = document.getElementById('semit');
const badgeEl  = document.getElementById('badgeEl');
const badgeWrap= document.getElementById('badgeWrap');
const cflash   = document.getElementById('cflash');
const colAlert = document.getElementById('colAlert');
const p1stat   = document.getElementById('p1stat');
const scenInfo = document.getElementById('scenInfo');

// ── TABS ─────────────────────────────────────────────────────────
function switchTab(name) {{
  // Deactivate all tabs
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('tab' + name).classList.add('active');

  // Hide all panes completely
  const paneSimu   = document.getElementById('paneSimu');
  const paneFisica = document.getElementById('paneFisica');
  const paneInterp = document.getElementById('paneInterp');
  paneSimu.style.display   = 'none';
  paneFisica.style.display = 'none';
  paneInterp.style.display = 'none';

  // Show the right one with the right display type
  if (name === 'Simu') {{
    paneSimu.style.display = 'grid';
    paneSimu.style.gridTemplateColumns = '200px 1fr 185px';
    paneSimu.style.flex = '1 1 auto';
    paneSimu.style.minHeight = '0';
    paneSimu.style.overflow = 'hidden';
  }} else if (name === 'Fisica') {{
    paneFisica.style.display = 'block';
    paneFisica.style.flex = '1 1 auto';
    paneFisica.style.overflowY = 'auto';
  }} else {{
    paneInterp.style.display = 'block';
    paneInterp.style.flex = '1 1 auto';
    paneInterp.style.overflowY = 'auto';
  }}
}}

// Init: show Simu pane
switchTab('Simu');

// ── SCENARIO CONFIG ───────────────────────────────────────────────
const SCEN = {{
  A:  {{ color:'var(--c)',  label:'MODO ONDA',       modo:'ONDA',      badge:'A',
         info:'<b>Cenário A — Interferência</b><br>Sem observação. Ψ=Ψ₁+Ψ₂. Padrão cos²·sinc².',
         infoClass:'info' }},
  B:  {{ color:'var(--cm)', label:'MODO PARTÍCULA',  modo:'COLAPSO',   badge:'B',
         info:'<b style="color:var(--cm)">Cenário B — Colapso Von Neumann</b><br>Detector ativo. P̂₁|Ψ⟩ colapsa o estado. Padrão sinc².',
         infoClass:'info om' }},
  D:  {{ color:'var(--g)',  label:'DECOERÊNCIA',     modo:'DECOER.',   badge:'D',
         info:'<b style="color:var(--g)">Cenário D — Decoerência</b><br>Acoplamento ambiental suprime interferência gradualmente.',
         infoClass:'info gn' }},
  DC: {{ color:'var(--pu)', label:'ESCOLHA ATRASADA',modo:'ATRASADA',  badge:'DC',
         info:'<b style="color:var(--pu)">Cenário DC — Escolha Atrasada</b><br>Decisão após passagem pela fenda. Wheeler (1978).',
         infoClass:'info pu' }},
}};

function updateScenarioUI() {{
  const s = SCEN[scenario];
  pbar.className = 'pbar ' + scenario;
  badgeEl.innerHTML = `<span class="badge ${{scenario}}">${{s.label}}</span>`;
  badgeWrap.innerHTML = `<span class="badge ${{scenario}}">${{s.label}}</span>`;
  smodo.textContent = s.modo;
  smodo.style.color = s.color;
  scenInfo.innerHTML = s.info;
  scenInfo.className = s.infoClass;
}}

// ── FRAME DISPLAY ─────────────────────────────────────────────────
function showFrame(i) {{
  const frames = FRAMES[scenario];
  if (!frames || frames.length === 0) return;
  i = Math.max(0, Math.min(i, frames.length - 1));
  curFrame = i;

  simImg.src = 'data:image/png;base64,' + frames[i];

  const pct   = (i + 1) / frames.length;
  const t_val = (i / frames.length) * NT_TOTAL * DT;
  pbar.style.width = (pct * 100) + '%';
  ovlTL.textContent = 't = ' + t_val.toFixed(3) + ' u.a.';
  st_el.textContent  = t_val.toFixed(3) + ' u.a.';
  sfn.textContent    = (i + 1) + '/' + frames.length;
  snorm.textContent  = (1 - pct * 0.07).toFixed(3);

  // Colapso flash (cenário B)
  if (scenario === 'B') {{
    const cf = Math.floor(frames.length * PRE_FRAC);
    if (i >= cf && !collapsed) {{
      collapsed = true;
      cflash.classList.remove('go');
      void cflash.offsetWidth;
      cflash.classList.add('go');
      colAlert.classList.add('show');
      setTimeout(() => colAlert.classList.remove('show'), 3800);
      sp1_el.textContent = (DET_PROB * 100).toFixed(2) + '%';
      p1stat.textContent = DET_PROB.toFixed(4);
    }}
  }}
}}

// ── PLAYBACK ──────────────────────────────────────────────────────
function play() {{
  if (playing) return;
  playing = true;
  document.getElementById('btnPlay').textContent = '⏸ PAUSE';
  timer = setInterval(() => {{
    const frames = FRAMES[scenario];
    if (curFrame < frames.length - 1) showFrame(curFrame + 1);
    else pause();
  }}, Math.max(25, 110 / speed));
}}

function pause() {{
  playing = false;
  clearInterval(timer);
  document.getElementById('btnPlay').textContent = '▶ PLAY';
}}

function resetSim() {{
  pause();
  curFrame  = 0;
  collapsed = false;
  colAlert.classList.remove('show');
  sp1_el.textContent = '—';
  showFrame(0);
}}

// ── CONTROLS ─────────────────────────────────────────────────────
document.getElementById('btnPlay').addEventListener('click', () => {{
  if (playing) pause(); else play();
}});
document.getElementById('btnReset').addEventListener('click', resetSim);
document.getElementById('btnStop').addEventListener('click', pause);

document.getElementById('spR').addEventListener('input', function() {{
  speed = +this.value;
  document.getElementById('spV').textContent = speed + '×';
  if (playing) {{ clearInterval(timer); play(); }}
}});

document.querySelectorAll('input[name=scen]').forEach(r => {{
  r.addEventListener('change', function() {{
    if (!this.checked) return;
    scenario = this.value;
    resetSim();
    updateScenarioUI();
  }});
}});

document.querySelectorAll('input[name=emit]').forEach(r => {{
  r.addEventListener('change', function() {{
    semit.textContent = this.value === 'single' ? 'INDIVIDUAL' : 'MÚLTIPLAS';
  }});
}});

document.getElementById('togDet').addEventListener('click', () => {{
  const tog = document.getElementById('togDet');
  const on  = !tog.classList.contains('on');
  tog.classList.toggle('on', on);
  const r = on ? 'B' : 'A';
  document.getElementById('r' + r).checked = true;
  scenario = r; resetSim(); updateScenarioUI();
}});

document.getElementById('togDelay').addEventListener('click', () => {{
  const tog = document.getElementById('togDelay');
  const on  = !tog.classList.contains('on');
  tog.classList.toggle('on', on);
  const r = on ? 'DC' : 'A';
  document.getElementById('r' + r).checked = true;
  scenario = r; resetSim(); updateScenarioUI();
}});

document.addEventListener('keydown', e => {{
  if (e.key === ' ')           {{ e.preventDefault(); if (playing) pause(); else play(); }}
  if (e.key === 'ArrowRight')  showFrame(curFrame + 1);
  if (e.key === 'ArrowLeft')   showFrame(curFrame - 1);
  if (e.key === 'r' || e.key === 'R') resetSim();
  if (e.key === '1') {{ document.getElementById('rA').checked  = true; scenario = 'A';  resetSim(); updateScenarioUI(); }}
  if (e.key === '2') {{ document.getElementById('rB').checked  = true; scenario = 'B';  resetSim(); updateScenarioUI(); }}
  if (e.key === '3') {{ document.getElementById('rD').checked  = true; scenario = 'D';  resetSim(); updateScenarioUI(); }}
  if (e.key === '4') {{ document.getElementById('rDC').checked = true; scenario = 'DC'; resetSim(); updateScenarioUI(); }}
}});

// ── INIT ──────────────────────────────────────────────────────────
updateScenarioUI();
showFrame(0);
</script>
</body>
</html>"""

# ══════════════════════════════════════════════════════════════════════════════
#  §6  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    console.print()
    console.print(Panel.fit(
        Text.assemble(
            ("╔═══════════════════════════════════════════════════════════╗\n", "bold cyan"),
            ("  DUPLA FENDA — TDSE 2D  |  4 Cenários  |  HTML Completo  \n", "bold white"),
            ("  Split-Operator | Von Neumann | Decoerência | Wheeler     \n", "dim cyan"),
            ("╚═══════════════════════════════════════════════════════════╝", "bold cyan"),
        ), border_style="cyan", padding=(0,2)
    ))

    # Grade (compartilhada entre cenários)
    with Progress(SpinnerColumn(style="cyan"),
                  TextColumn("[cyan]{task.description}"),
                  TimeElapsedColumn(), console=console) as prog:
        t = prog.add_task("Construindo grade 2D e barreira...", total=None)
        x, y, X, Y, KX, KY = build_grids()
        V, ab, xw, iw, slit1, slit2 = build_barrier(y)
        prog.update(t, description="[green]✓ Grade 320×160 + barreira prontas")

    tbl = Table(box=box.SIMPLE_HEAVY, style="dim")
    tbl.add_column("Parâmetro", style="cyan")
    tbl.add_column("Valor", justify="right")
    for k, v in [("Grade",f"{p.Nx}×{p.Ny}"),("dt / Passos",f"{p.dt} / {p.Nt_pre+p.Nt_post}"),
                  ("λ",f"{p.lam:.3f} u.a."),("k₀",str(p.k0)),("d/λ",str(p.slit_d)),
                  ("a/λ",str(p.slit_a)),("V₀",str(int(p.V0))),("Frames/cen.",str(p.N_FRAMES))]:
        tbl.add_row(k, v)
    console.print(tbl)

    t_total = time.time()

    console.print()
    console.print(Rule("[bold cyan]Cenário A — Interferência Quântica[/bold cyan]", style="cyan"))
    t0=time.time(); frA, psiA, _ = run_scenario("A",x,y,X,Y,KX,KY,slit1,slit2,xw,iw,"A")
    profA = screen_profile(psiA); console.print(f"[dim]  {time.time()-t0:.1f}s | {len(frA)} frames[/dim]")

    console.print()
    console.print(Rule("[bold red]Cenário B — Colapso Von Neumann[/bold red]", style="red"))
    t0=time.time(); frB, psiB, dp = run_scenario("B",x,y,X,Y,KX,KY,slit1,slit2,xw,iw,"B")
    profB = screen_profile(psiB); console.print(f"[dim]  {time.time()-t0:.1f}s | {len(frB)} frames[/dim]")

    console.print()
    console.print(Rule("[bold green]Cenário D — Decoerência Quântica[/bold green]", style="green"))
    t0=time.time(); frD, psiD, _ = run_scenario("D",x,y,X,Y,KX,KY,slit1,slit2,xw,iw,"D",decoherence_strength=0.9)
    profD = screen_profile(psiD); console.print(f"[dim]  {time.time()-t0:.1f}s | {len(frD)} frames[/dim]")

    console.print()
    console.print(Rule("[bold magenta]Cenário DC — Escolha Atrasada (Wheeler)[/bold magenta]", style="magenta"))
    t0=time.time(); frDC, _, _ = run_scenario("DC",x,y,X,Y,KX,KY,slit1,slit2,xw,iw,"DC")
    console.print(f"[dim]  {time.time()-t0:.1f}s | {len(frDC)} frames[/dim]")

    console.print()
    console.print("[cyan]Renderizando perfil comparativo I(y)...[/cyan]")
    profB64 = render_profile_comparison(profA, profB, profD, y, det_prob=dp)

    console.print("[cyan]Montando HTML...[/cyan]")
    Nt_pre_frac = p.Nt_pre / (p.Nt_pre + p.Nt_post)
    html = build_html(frA, frB, frD, frDC, profB64, dp, Nt_pre_frac)

    out = "double_slit_full.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)

    sz = os.path.getsize(out) / 1024 / 1024
    elapsed = time.time() - t_total

    console.print()
    console.print(Panel(
        Text.assemble(
            ("[bold green]✓ SIMULAÇÃO COMPLETA![/bold green]\n\n",""),
            (f"  Arquivo: [cyan]{os.path.abspath(out)}[/cyan]\n",""),
            (f"  Tamanho: [yellow]{sz:.1f} MB[/yellow]\n",""),
            (f"  Frames:  {len(frA)}A + {len(frB)}B + {len(frD)}D + {len(frDC)}DC\n","dim white"),
            (f"  Tempo total: [white]{elapsed:.1f}s[/white]\n\n",""),
            ("  Abra o arquivo HTML no navegador.\n\n","dim"),
            ("  Controles:\n","bold white"),
            ("  ESPAÇO play/pause · ← → frame a frame · R reiniciar\n","dim"),
            ("  1/2/3/4 trocar cenário · 3 abas funcionais\n","dim"),
        ),
        border_style="green", padding=(1,2)
    ))


if __name__ == "__main__":
    main()