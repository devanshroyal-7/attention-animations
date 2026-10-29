"""
Attention from scratch -- Manim CE scenes for 11-785 Lab 10 (F26).

Styling matches the deck: white background, Arial-style sans font,
Q = orange (#FFAB40), K = green (#6AA84F), S / scores = olive (#B5AA48),
negative values = purple. Cell shade intensity encodes |value|.

Render (1080p, 30 fps):
    SEED=31 manim -r 1920,1080 --fps 30 attention_scenes.py SceneA_DotProduct SceneB_Softmax SceneC_Matrix
Env vars: SEED (default 31), D (dim, default 4), N (tokens/keys, default 4),
          SCALE (1 = divide scores by sqrt(d_k) like the deck, 0 = plain softmax).
"""
import os
import numpy as np
from manim import *

# ---------------------------------------------------------------- data
SEED = int(os.environ.get("SEED", 31))
D = int(os.environ.get("D", 4))
N = int(os.environ.get("N", 4))
SCALE = os.environ.get("SCALE", "1") == "1"

rng = np.random.default_rng(SEED)
Q = np.round(rng.normal(0, 1, (N, D)), 1)   # queries, one row per token
K = np.round(rng.normal(0, 1, (N, D)), 1)   # keys, one row per token
q = Q[0]                                      # the query used in scenes A/B
S_MAT = Q @ K.T                               # raw similarity scores
DIV = np.sqrt(D) if SCALE else 1.0
S_SCALED = S_MAT / DIV
E_MAT = np.exp(S_SCALED - S_SCALED.max(axis=1, keepdims=True))
A_MAT = E_MAT / E_MAT.sum(axis=1, keepdims=True)  # attention weights
s = S_MAT[0]
s_sc = S_SCALED[0]
e = np.exp(s_sc)        # un-shifted exp for display (small numbers anyway)
a = e / e.sum()

# ---------------------------------------------------------------- style
config.background_color = WHITE
FONT = "Arimo"  # metric-compatible with Arial (the deck font)
INK = "#222222"
Q_COL = "#FFAB40"
K_COL = "#6AA84F"
S_COL = "#B5AA48"
NEG_COL = "#8E7CC3"
HI_COL = "#E06666"
CELL = 0.72

def darker(c, f=0.45):
    return interpolate_color(ManimColor(c), ManimColor("#000000"), f)

def shade(v, vmax, col):
    """white -> deck color -> darker, by |v|/vmax; negatives use purple."""
    base = col if v >= 0 else NEG_COL
    t = min(abs(v) / max(vmax, 1e-9), 1.0)
    if t < 0.7:
        return interpolate_color(ManimColor(WHITE), ManimColor(base), 0.12 + t / 0.7 * 0.88)
    return interpolate_color(ManimColor(base), darker(base), (t - 0.7) / 0.3)

def lum(c):
    r, g, b = ManimColor(c).to_rgb()
    return 0.299 * r + 0.587 * g + 0.114 * b

def T(txt, size=28, color=INK, **kw):
    return Text(txt, font=FONT, font_size=size, color=color, **kw)

def M(markup, size=28, color=INK):
    return MarkupText(markup, font=FONT, font_size=size, color=color)

def fmt(v, nd=1):
    return f"{v:.{nd}f}".replace("-0.00", "0.00") if nd else f"{v:.0f}"

def cell(v, vmax, col, nd=1, size=CELL, fs=20):
    fill = shade(v, vmax, col)
    sq = Square(side_length=size, stroke_color="#555555", stroke_width=1.5,
                fill_color=fill, fill_opacity=1)
    num = T(fmt(v, nd), fs, color=WHITE if lum(fill) < 0.45 else INK)
    num.move_to(sq)
    return VGroup(sq, num)

def vec_row(vals, vmax, col, nd=1, size=CELL, fs=20):
    return VGroup(*[cell(v, vmax, col, nd, size, fs) for v in vals]).arrange(RIGHT, buff=0)

def mat(M_, vmax, col, nd=1, size=CELL, fs=20):
    rows = VGroup(*[vec_row(r, vmax, col, nd, size, fs) for r in M_])
    return rows.arrange(DOWN, buff=0)

def shape_tag(mob, txt):
    return T(txt, 16, color="#555555").next_to(mob, DR, buff=0.05).shift(UP * 0.15)

def legend(items):
    """items: list of (color, label); purple = negative is always appended."""
    sw = lambda col: Square(0.28, stroke_color="#555555", stroke_width=1, fill_color=col, fill_opacity=1)
    parts = []
    for col, name in list(items) + [(NEG_COL, "negative")]:
        parts += [sw(col), T(name, 16, "#555555")]
    parts.append(T("(darker = larger |value|)", 16, "#555555"))
    return VGroup(*parts).arrange(RIGHT, buff=0.15).to_corner(DL, buff=0.35)

def title(txt):
    return T(txt, 40).to_corner(UL, buff=0.6)

VMAX_QK = max(np.abs(Q).max(), np.abs(K).max())
VMAX_S = np.abs(S_MAT).max()


# ================================================================ Scene A
class SceneA_DotProduct(Scene):
    """One query vs each key: elementwise product, sum -> similarity score."""
    def construct(self):
        ttl = title("Similarity Score")
        self.add(ttl)
        lg = legend([(Q_COL, "query"), (K_COL, "key"), (S_COL, "score")])
        self.add(lg)

        qv = vec_row(q, VMAX_QK, Q_COL)
        q_lab = M("Q<sub>1</sub>", 30)
        keys = VGroup(*[vec_row(K[i], VMAX_QK, K_COL) for i in range(N)]).arrange(DOWN, buff=0.28)
        left = VGroup(qv, keys).arrange(DOWN, buff=0.7).move_to(LEFT * 4.3 + DOWN * 0.3)
        q_lab.next_to(qv, LEFT, buff=0.3)
        k_labs = VGroup(*[M(f"K<sub>{i+1}</sub>", 28).next_to(keys[i], LEFT, buff=0.3) for i in range(N)])

        self.play(FadeIn(qv, shift=RIGHT * 0.3), Write(q_lab), run_time=1)
        self.play(LaggedStart(*[FadeIn(k, shift=RIGHT * 0.3) for k in keys], lag_ratio=0.2),
                  FadeIn(k_labs), run_time=1.5)
        self.wait(0.5)

        # score row (bottom right), initially empty slots
        slots = VGroup(*[Square(CELL, stroke_color="#999999", stroke_width=1.5) for _ in range(N)])
        slots.arrange(RIGHT, buff=0).move_to(RIGHT * 2.6 + DOWN * 2.7)
        s_lab = M("S", 30).next_to(slots, LEFT, buff=0.3)
        s_tags = VGroup(*[M(f"s<sub>{i+1}</sub>", 18, "#555555").next_to(slots[i], DOWN, buff=0.1) for i in range(N)])
        self.play(Create(slots), Write(s_lab), FadeIn(s_tags), run_time=0.8)

        work_c = RIGHT * 2.6 + UP * 1.4
        score_cells = []
        for i in range(N):
            fast = i > 0
            rt = 0.6 if fast else 1.0
            hl = SurroundingRectangle(keys[i], color=HI_COL, buff=0.06, stroke_width=3)
            qc = qv.copy()
            kc = keys[i].copy()
            self.play(Create(hl), run_time=0.3 if fast else 0.5)
            self.play(qc.animate.move_to(work_c), kc.animate.move_to(work_c + DOWN * (CELL + 0.25)), run_time=rt)
            op = T("×", 30).next_to(VGroup(qc, kc), LEFT, buff=0.3)
            q_lab2 = M("Q<sub>1</sub>", 26).next_to(qc, RIGHT, buff=0.3)
            k_lab2 = M(f"K<sub>{i+1}</sub>", 26).next_to(kc, RIGHT, buff=0.3)
            self.play(FadeIn(op), FadeIn(q_lab2), FadeIn(k_lab2), run_time=0.3)

            prods = q * K[i]
            pv = vec_row(prods, VMAX_QK ** 2, S_COL, nd=2, fs=17)
            pv.move_to(kc).shift(DOWN * (CELL + 0.55))
            eq_line = Line(kc.get_left() + DOWN * 0.5, kc.get_right() + DOWN * 0.5, color=INK, stroke_width=2)
            anims = [TransformFromCopy(VGroup(qc[j], kc[j]), pv[j]) for j in range(D)]
            self.play(Create(eq_line), LaggedStart(*anims, lag_ratio=0.15), run_time=rt * 1.4)
            if not fast:
                self.wait(0.5)

            # sum
            total = cell(s[i], VMAX_S, S_COL, nd=2, fs=17)
            sig = T("Σ", 34).next_to(pv, RIGHT, buff=0.35)
            total.next_to(sig, RIGHT, buff=0.35)
            self.play(FadeIn(sig), run_time=0.3)
            self.play(ReplacementTransform(pv.copy(), total), run_time=rt)
            if not fast:
                expl = T("bigger = more similar", 20, "#555555").next_to(total, DOWN, buff=0.25)
                self.play(FadeIn(expl), run_time=0.4)
                self.wait(1.0)
                self.play(FadeOut(expl), run_time=0.3)
            self.play(total.animate.move_to(slots[i]), run_time=rt)
            score_cells.append(total)
            self.play(FadeOut(VGroup(qc, kc, op, q_lab2, k_lab2, pv, eq_line, sig, hl)), run_time=0.4)

        best = int(np.argmax(s))
        box1 = SurroundingRectangle(score_cells[best], color=HI_COL, buff=0.05, stroke_width=4)
        box2 = SurroundingRectangle(VGroup(keys[best], k_labs[best]), color=HI_COL, buff=0.06, stroke_width=4)
        note = M(f"Q<sub>1</sub> is most similar to K<sub>{best+1}</sub>", 26).next_to(slots, UP, buff=0.6)
        self.play(Create(box1), Create(box2), Write(note), run_time=1)
        self.wait(2.5)


# ================================================================ Scene B
class SceneB_Softmax(Scene):
    """Scores -> (÷√d_k) -> exp -> normalize -> attention weights that sum to 1."""
    def construct(self):
        ttl = title("Similarity Score → Attention Weights")
        self.add(ttl)
        X0 = LEFT * 1.0
        base_y = -0.4

        def row_with_label(vals, vmax, col, nd, label_markup, y):
            r = vec_row(vals, vmax, col, nd=nd, fs=18).move_to(X0 + UP * y)
            lab = M(label_markup, 26).next_to(r, LEFT, buff=0.35)
            return r, lab

        def bars(vals, unit, col, anchor_row, y0):
            bs = VGroup()
            for j, v in enumerate(vals):
                h = max(abs(v) * unit, 0.02)
                b = Rectangle(width=CELL * 0.7, height=h, stroke_width=0,
                              fill_color=col if v >= 0 else NEG_COL, fill_opacity=0.9)
                x = anchor_row[j].get_center()[0]
                b.move_to([x, y0 + (h / 2 if v >= 0 else -h / 2), 0])
                bs.add(b)
            return bs

        # keys labels row
        srow, slab = row_with_label(s, VMAX_S, S_COL, 2, "S", base_y)
        k_tags = VGroup(*[M(f"K<sub>{i+1}</sub>", 20, "#555555").next_to(srow[i], DOWN, buff=0.12) for i in range(N)])
        self.play(FadeIn(srow), Write(slab), FadeIn(k_tags), run_time=1)

        # bar chart on the right
        chart_x = RIGHT * 4.6
        axis = Line(chart_x + LEFT * 1.8 + DOWN * 0.9, chart_x + RIGHT * 1.8 + DOWN * 0.9, color=INK, stroke_width=2)
        anchors = VGroup(*[Dot(chart_x + LEFT * 1.8 + RIGHT * (0.45 + 0.9 * j) + DOWN * 0.9, radius=0.001) for j in range(N)])
        y0 = axis.get_center()[1]
        unit_s = 1.0 / max(np.abs(s).max(), 1e-9)
        neg_h = max(-s.min(), 0) * unit_s
        b_tags = VGroup(*[M(f"K<sub>{j+1}</sub>", 18, "#555555").next_to(anchors[j], DOWN, buff=0.15 + neg_h) for j in range(N)])
        cur_bars = bars(s, unit_s, S_COL, anchors, y0)
        chart_lab = T("scores", 22, "#555555").next_to(axis, UP, buff=2.2)
        self.play(Create(axis), FadeIn(b_tags), GrowFromEdge(cur_bars, DOWN), FadeIn(chart_lab), run_time=1)
        self.wait(0.6)

        cur_row, cur_lab = srow, slab
        if SCALE:
            div = M(f"÷ √d<sub>k</sub> = ÷ {DIV:g}", 24).next_to(srow, RIGHT, buff=0.4)
            self.play(Write(div), run_time=0.7)
            r2, l2 = row_with_label(s_sc, VMAX_S / DIV, S_COL, 2, f"S / √d<sub>k</sub>", base_y)
            self.play(FadeOut(div), Transform(cur_row, r2), Transform(cur_lab, l2), run_time=1)
            self.wait(0.6)

        # exp
        exp_note = M("Step 1: e<sup>x</sup> makes every score positive", 24).move_to(X0 + UP * 1.9)
        self.play(FadeIn(exp_note), run_time=0.5)
        erow, elab = row_with_label(e, e.max(), S_COL, 2, "e<sup>x</sup>", base_y - 1.4)
        self.play(TransformFromCopy(cur_row, erow), Write(elab), run_time=1.2)
        unit_e = 1.6 / e.max()
        e_bars = bars(e, unit_e, S_COL, anchors, y0)
        new_lab = T("e^x", 22, "#555555").move_to(chart_lab)
        self.play(Transform(cur_bars, e_bars), Transform(chart_lab, new_lab), run_time=1)
        tot = M(f"Σ = {e.sum():.2f}", 24).next_to(erow, RIGHT, buff=0.4)
        self.play(Write(tot), run_time=0.7)
        self.wait(0.8)

        # normalize
        norm_note = M("Step 2: divide by the sum  →  weights add up to 1", 24).move_to(exp_note)
        self.play(Transform(exp_note, norm_note), run_time=0.6)
        arow, alab = row_with_label(a, 1.0, S_COL, 2, "softmax", base_y - 2.8)
        alab.next_to(arow, LEFT, buff=0.35)
        self.play(TransformFromCopy(erow, arow), Write(alab), run_time=1.2)
        unit_a = 1.6 / a.max()
        a_bars = bars(a, unit_a, S_COL, anchors, y0)
        a_nums = VGroup(*[T(f"{a[j]:.2f}", 16).next_to(a_bars[j], UP, buff=0.06) for j in range(N)])
        new_lab = T("attention weights", 22, "#555555").move_to(chart_lab)
        self.play(Transform(cur_bars, a_bars), Transform(chart_lab, new_lab), FadeIn(a_nums), run_time=1)
        sum1 = M(f"Σ = {a.sum():.2f}", 24, HI_COL).next_to(arow, RIGHT, buff=0.4)
        self.play(Write(sum1), run_time=0.7)
        self.wait(0.6)

        best = int(np.argmax(a))
        hb = SurroundingRectangle(VGroup(cur_bars[best], a_nums[best]), color=HI_COL, buff=0.08, stroke_width=4)
        hc = SurroundingRectangle(VGroup(srow[best]), color=HI_COL, buff=0.05, stroke_width=4)
        hc2 = SurroundingRectangle(arow[best], color=HI_COL, buff=0.05, stroke_width=4)
        fin = M(f"Most similar key (K<sub>{best+1}</sub>) gets the most attention: {a[best]:.0%}", 24, HI_COL)
        fin.move_to(exp_note)
        self.play(Create(hb), Create(hc), Create(hc2), Transform(exp_note, fin), run_time=1)
        self.wait(3)


# ================================================================ Scene C
class SceneC_Matrix(Scene):
    """Stack queries into Q and keys into K; S = QK^T; row-wise softmax -> attention matrix."""
    def construct(self):
        ttl = title("Matrix Similarity Score")
        self.add(ttl)
        cs = 0.62
        fs = 16

        # Q and K as stacked rows
        Qm = mat(Q, VMAX_QK, Q_COL, size=cs, fs=fs)
        Km = mat(K, VMAX_QK, K_COL, size=cs, fs=fs)
        VGroup(Qm, Km).arrange(RIGHT, buff=2.2).move_to(LEFT * 2.5 + DOWN * 0.2)
        q_rows_lab = VGroup(*[M(f"Q<sub>{i+1}</sub>", 18).next_to(Qm[i], LEFT, buff=0.15) for i in range(N)])
        k_rows_lab = VGroup(*[M(f"K<sub>{i+1}</sub>", 18).next_to(Km[i], LEFT, buff=0.15) for i in range(N)])
        Ql = M("Q", 30).next_to(Qm, DOWN, buff=0.45)
        Kl = M("K", 30).next_to(Km, DOWN, buff=0.45)
        Qt = shape_tag(Qm, f"{N}x{D}")
        Kt = shape_tag(Km, f"{N}x{D}")

        # build by stacking rows one at a time
        for i in range(N):
            self.play(FadeIn(Qm[i], shift=DOWN * 0.2), FadeIn(q_rows_lab[i]),
                      FadeIn(Km[i], shift=DOWN * 0.2), FadeIn(k_rows_lab[i]), run_time=0.45)
        self.play(Write(Ql), Write(Kl), FadeIn(Qt), FadeIn(Kt), run_time=0.6)
        self.wait(0.6)

        # transpose K -> K^T: row i becomes column i
        KT = VGroup()
        for i in range(N):
            col = VGroup(*[cell(K[i][j], VMAX_QK, K_COL, size=cs, fs=fs) for j in range(D)]).arrange(DOWN, buff=0)
            KT.add(col)
        KT.arrange(RIGHT, buff=0).move_to(Km)
        kt_cols_lab = VGroup(*[M(f"K<sub>{i+1}</sub>", 18).next_to(KT[i], UP, buff=0.12) for i in range(N)])
        KTl = M("K<sup>T</sup>", 30).move_to(Kl)
        KTt = shape_tag(KT, f"{D}x{N}")
        self.play(*[ReplacementTransform(Km[i], KT[i]) for i in range(N)],
                  ReplacementTransform(k_rows_lab, kt_cols_lab),
                  Transform(Kl, KTl), Transform(Kt, KTt), run_time=1.6)
        at = T("@", 32).move_to((Qm.get_right() + KT.get_left()) / 2)
        self.play(FadeIn(at), run_time=0.4)
        self.wait(0.5)

        # S = Q K^T
        Sm = mat(S_MAT, VMAX_S, S_COL, nd=2, size=cs, fs=14)
        Sm.next_to(KT, RIGHT, buff=1.4)
        eq = T("=", 32).move_to((KT.get_right() + Sm.get_left()) / 2)
        Sl = M("S", 30).next_to(Sm, DOWN, buff=0.45).align_to(Ql, DOWN)
        St = shape_tag(Sm, f"{N}x{N}")
        self.play(FadeIn(eq), FadeIn(Sl), FadeIn(St), run_time=0.4)

        # first entry slowly: row 1 of Q with column j of K^T
        for i in range(N):
            for j in range(N):
                slow = (i == 0)
                if slow:
                    r1 = SurroundingRectangle(Qm[i], color=HI_COL, buff=0.03, stroke_width=3)
                    r2 = SurroundingRectangle(KT[j], color=HI_COL, buff=0.03, stroke_width=3)
                    self.play(Create(r1), Create(r2), run_time=0.25)
                    self.play(TransformFromCopy(VGroup(Qm[i], KT[j]), Sm[i][j]), run_time=0.5)
                    self.remove(r1, r2)
                else:
                    pass
            if i == 0:
                note = M("row 1 = the scores from one query (Q<sub>1</sub>)", 20, HI_COL).next_to(Sm, UP, buff=0.25)
                self.play(FadeIn(note), run_time=0.4)
                self.wait(1.0)
                self.play(FadeOut(note), run_time=0.3)
            else:
                rq = SurroundingRectangle(Qm[i], color=HI_COL, buff=0.03, stroke_width=3)
                self.play(Create(rq), run_time=0.15)
                self.play(*[TransformFromCopy(VGroup(Qm[i], KT[j]), Sm[i][j]) for j in range(N)], run_time=0.6)
                self.remove(rq)
        self.wait(0.8)

        # clear left side, move S to center-left, softmax row-wise
        left_stuff = VGroup(Qm, q_rows_lab, Ql, Qt, at, KT, kt_cols_lab, Kl, Kt, eq)
        self.play(FadeOut(left_stuff), run_time=0.6)
        self.play(VGroup(Sm, Sl, St).animate.move_to(LEFT * 3.6 + DOWN * 0.2), run_time=0.8)
        s_row_lab = VGroup(*[M(f"Q<sub>{i+1}</sub>", 18).next_to(Sm[i], LEFT, buff=0.15) for i in range(N)])
        s_col_lab = VGroup(*[M(f"K<sub>{j+1}</sub>", 18).next_to(Sm[0][j], UP, buff=0.12) for j in range(N)])
        self.play(FadeIn(s_row_lab), FadeIn(s_col_lab), run_time=0.4)

        op_txt = f"softmax( S / √d<sub>k</sub> )" if SCALE else "softmax( S )"
        arrow = Arrow(Sm.get_right() + RIGHT * 0.5, Sm.get_right() + RIGHT * 3.1, color=INK, stroke_width=3, buff=0)
        op = M(op_txt, 20).next_to(arrow, UP, buff=0.15)
        rowwise = T("row by row", 18, "#555555").next_to(arrow, DOWN, buff=0.15)
        self.play(GrowArrow(arrow), Write(op), FadeIn(rowwise), run_time=0.9)

        Am = mat(A_MAT, 1.0, S_COL, nd=2, size=cs, fs=14).next_to(arrow, RIGHT, buff=0.8)
        Al = M("Attention", 30).next_to(Am, DOWN, buff=0.45).align_to(Sl, DOWN)
        a_row_lab = VGroup(*[M(f"Q<sub>{i+1}</sub>", 18).next_to(Am[i], LEFT, buff=0.15) for i in range(N)])
        a_col_lab = VGroup(*[M(f"K<sub>{j+1}</sub>", 18).next_to(Am[0][j], UP, buff=0.12) for j in range(N)])
        sums = VGroup(*[M(f"Σ = {A_MAT[i].sum():.2f}", 18, HI_COL).next_to(Am[i], RIGHT, buff=0.3) for i in range(N)])
        At = shape_tag(Am, f"{N}x{N}")
        self.play(FadeIn(Al), FadeIn(a_col_lab), FadeIn(At), run_time=0.4)
        for i in range(N):
            hl = SurroundingRectangle(Sm[i], color=HI_COL, buff=0.03, stroke_width=3)
            self.play(Create(hl), run_time=0.2)
            self.play(TransformFromCopy(Sm[i], Am[i]), FadeIn(a_row_lab[i]), run_time=0.6)
            self.play(FadeIn(sums[i]), run_time=0.25)
            self.remove(hl)

        boxes = VGroup(*[SurroundingRectangle(Am[i][int(np.argmax(A_MAT[i]))], color=HI_COL, buff=0.03, stroke_width=4) for i in range(N)])
        fin = T("Each row: how much one query attends to every key", 24, INK).to_edge(DOWN, buff=0.5)
        self.play(Create(boxes), FadeIn(fin), run_time=1)
        self.wait(3)
