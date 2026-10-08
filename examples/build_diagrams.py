"""Generate implementation diagrams using the existing Matplotlib dependency."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

ROOT = Path(__file__).resolve().parents[1]
BLUE, TEAL, ORANGE, RED = "#25577a", "#176b63", "#9c5c19", "#a34444"


def figure(width=8, height=4.8):
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    fig.subplots_adjust(left=.02, right=.98, bottom=.02, top=.98)
    return fig, ax


def box(ax, x, y, text, w=.28, h=.1, color=BLUE, size=9):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                               linewidth=1.2, edgecolor=color, facecolor="#f2f6f8", zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color="#192f3b", linespacing=1.25, zorder=4)


def arrow(ax, start, end, text=None, *, color=BLUE, dashed=False, both=False):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="<->" if both else "-|>", mutation_scale=11,
                                linewidth=1.2, color=color, linestyle="--" if dashed else "-"))
    if text:
        ax.text((start[0]+end[0])/2+.015, (start[1]+end[1])/2+.02, text, fontsize=8,
                color=color, ha="center", va="center", bbox=dict(facecolor="white", edgecolor="none", pad=1))


def decision(ax, x, y, text, w=.25, h=.12):
    ax.add_patch(Polygon([(x-w/2,y),(x,y+h/2),(x+w/2,y),(x,y-h/2)], closed=True,
                         facecolor="#fdf4e6", edgecolor=ORANGE, linewidth=1.2))
    ax.text(x, y, text, ha="center", va="center", fontsize=9, color="#192f3b")


def save(fig, name):
    target = ROOT / "docs/diagrams"
    target.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "pdf", "png"):
        output = target / f"{name}.{extension}"
        fig.savefig(output, dpi=300, facecolor="white")
        if extension == "svg":
            # Matplotlib emits trailing spaces in multiline path attributes.
            content = output.read_text(encoding="utf-8")
            output.write_text("\n".join(line.rstrip() for line in content.splitlines()) + "\n",
                              encoding="utf-8", newline="\n")
    plt.close(fig)


def astar_workflow():
    fig, ax = figure(7.6, 5)
    box(ax, .44, .94, "Start: static grid, start, goal", w=.39, h=.07)
    box(ax, .44, .82, "Initialize frontier\ng(start)=0; f=g+h; FIFO ties", w=.39, h=.09)
    box(ax, .44, .68, "Pop lowest-f live state\nDiscard stale heap entries", w=.39, h=.09)
    decision(ax, .44, .54, "Goal?", w=.22, h=.11)
    box(ax, .79, .54, "Reconstruct path\nfrom parents", w=.24, h=.1, color=TEAL)
    box(ax, .79, .36, "Return optimal cost\nand search metrics", w=.24, h=.1, color=TEAL)
    box(ax, .44, .4, "Generate free orthogonal neighbors", w=.39, h=.075)
    box(ax, .44, .27, "If g improves: update parent and g\nCompute h/f; insert or reopen", w=.41, h=.095)
    decision(ax, .44, .13, "Frontier nonempty?", w=.32, h=.1)
    box(ax, .79, .13, "Unreachable\nNo path/cost", w=.24, h=.09, color=RED)
    for a,b in ((.905,.87),(.77,.73),(.635,.595),(.485,.445),(.36,.32),(.22,.18)):
        arrow(ax,(.44,a),(.44,b))
    arrow(ax,(.55,.54),(.66,.54),"yes",color=TEAL)
    arrow(ax,(.79,.48),(.79,.42),color=TEAL)
    ax.text(.48,.465,"no",fontsize=8,color=ORANGE)
    arrow(ax,(.61,.13),(.66,.13),"no",color=RED)
    arrow(ax,(.27,.13),(.1,.13),"yes")
    arrow(ax,(.1,.13),(.1,.68))
    arrow(ax,(.1,.68),(.235,.68))
    save(fig,"astar_workflow")


def centralized_workflow():
    fig, ax = figure(9, 4.5)
    for x,text in ((.17,"Independent Manhattan A*\npaths and costs L_i"),(.5,"Predicted vertex/edge events\nC_i and low-degree exposure B_i"),(.83,"Initial order\n(-C_i, -B_i, -L_i, ID)")):
        box(ax,x,.86,text,w=.29,h=.15)
    arrow(ax,(.323,.86),(.347,.86))
    arrow(ax,(.653,.86),(.677,.86))
    arrow(ax,(.83,.775),(.83,.64))
    box(ax,.83,.55,"Next robot: Space-Time A*\nOwn start -> goal, moves + WAIT\nGrow horizon within finite cap",w=.29,h=.17)
    box(ax,.5,.55,"Successful trajectory\nCommit vertices, edges\nand permanent goal hold",w=.28,h=.17,color=TEAL)
    box(ax,.17,.55,"All robots planned?\nIndependent collision check\nThen execute verified paths",w=.28,h=.17,color=TEAL)
    arrow(ax,(.677,.55),(.648,.55),"found",color=TEAL)
    arrow(ax,(.352,.55),(.318,.55),color=TEAL)
    arrow(ax,(.5,.645),(.5,.705))
    arrow(ax,(.5,.705),(.83,.705),"more robots")
    arrow(ax,(.83,.705),(.83,.645))
    box(ax,.83,.24,"Order failure\nEligible promotion remains?",w=.29,h=.12,color=ORANGE)
    box(ax,.5,.24,"Promote first failed robot\nSkip previously tried orders",w=.28,h=.12,color=ORANGE)
    box(ax,.17,.24,"Clear paths and reservations\nRestart whole order",w=.28,h=.12,color=ORANGE)
    arrow(ax,(.83,.455),(.83,.31),"failure",color=ORANGE,dashed=True)
    arrow(ax,(.678,.24),(.648,.24),"yes",color=ORANGE,dashed=True)
    arrow(ax,(.352,.24),(.318,.24),color=ORANGE,dashed=True)
    arrow(ax,(.17,.17),(.17,.08),color=ORANGE,dashed=True)
    arrow(ax,(.17,.08),(.994,.08),color=ORANGE,dashed=True)
    arrow(ax,(.994,.08),(.994,.55),color=ORANGE,dashed=True)
    arrow(ax,(.994,.55),(.978,.55),color=ORANGE,dashed=True)
    arrow(ax,(.83,.171),(.83,.105),"no",color=RED,dashed=True)
    ax.text(.61,.03,"No eligible order or exhausted budget: bounded failure (no execution)",
            fontsize=8,color=RED,ha="center")
    save(fig,"centralized_cg_workflow")


def architecture():
    fig, ax = figure(9, 4.5)
    ax.text(.25,.97,"CENTRALIZED",ha="center",fontsize=12,color=BLUE,weight="bold")
    ax.text(.75,.97,"DECENTRALIZED SIMULATION",ha="center",fontsize=12,color=TEAL,weight="bold")
    ax.plot([.5,.5],[.05,.95],color="#b8c6ce",lw=1)
    box(ax,.25,.81,"Common static obstacle map",w=.4,h=.09)
    box(ax,.75,.81,"Common static obstacle map",w=.4,h=.09,color=TEAL)
    box(ax,.25,.54,"One CooperativePlanner\nGlobal priority order\nOne reservation table\nSequential reserved searches",w=.4,h=.27)
    arrow(ax,(.25,.758),(.25,.683))
    for x,label in ((.1,"Path A1"),(.25,"Path A2"),(.4,"Path A3")):
        box(ax,x,.24,label,w=.12,h=.09)
        arrow(ax,(.25,.397),(x,.293))
    for x,label in ((.635,"Controller A1"),(.865,"Controller A2")):
        box(ax,x,.55,label+"\nLocal A*/ST-A*\nOwn reservation table\nReceived peer proposals",w=.185,h=.27,color=TEAL,size=8.4)
        arrow(ax,(x,.758),(x,.69),color=TEAL)
    arrow(ax,(.736,.55),(.764,.55),color=TEAL,both=True)
    ax.text(.75,.145,"Immutable peer messages (one per receiver)",ha="center",fontsize=8.5,color=TEAL)
    box(ax,.75,.24,"Neutral scheduler: delivery + round barriers\nNo priority assignment or path computation",w=.43,h=.12,color=TEAL,size=8.5)
    arrow(ax,(.63,.305),(.63,.405),color=TEAL,dashed=True)
    arrow(ax,(.87,.305),(.87,.405),color=TEAL,dashed=True)
    ax.text(.75,.065,"Final collision observer may reject, never repair.\nControllers use messages, not peers' mutable state.",
            ha="center",fontsize=8.5,color="#192f3b",linespacing=1.4)
    save(fig,"centralized_vs_decentralized")


def negotiation_sequence():
    fig, ax = figure(9, 5.4)
    box(ax,.34,.94,"RobotController A1",w=.27,h=.065,color=TEAL)
    box(ax,.81,.94,"RobotController A2",w=.27,h=.065,color=TEAL)
    for x in (.34,.81):
        ax.plot([x,x],[.12,.9],ls="--",lw=1,color="#a9b9c2",zorder=0)
    for y,label in ((.83,"R0"),(.745,"R1"),(.50,"R2"),(.275,"R3")):
        ax.text(.045,y,label,fontsize=10,weight="bold",color=BLUE)
    box(ax,.34,.83,"Independent A* -> proposal v1",w=.35,h=.07,size=8.5)
    box(ax,.81,.83,"Independent A* -> proposal v1",w=.35,h=.07,size=8.5)
    arrow(ax,(.34,.755),(.81,.755),"PATH_PROPOSAL delivered",color=TEAL)
    arrow(ax,(.81,.695),(.34,.695),"PATH_PROPOSAL delivered",color=TEAL)
    box(ax,.34,.61,"Local conflict / key comparison\nA1 retains proposal",w=.35,h=.1,size=8.5)
    box(ax,.81,.61,"Same pair decision: A2 yields\nLocal ST-A*, own table -> v2",w=.35,h=.1,color=ORANGE,size=8.5)
    arrow(ax,(.34,.51),(.81,.51),"Previous notices + decisions",color=TEAL)
    arrow(ax,(.81,.455),(.34,.455),"Notices + decisions + UPDATED_PATH v2",color=TEAL)
    box(ax,.575,.37,"Both locally verify compatible proposals\nQueue AGREEMENT vector ((1,1), (2,2))",w=.76,h=.09,color=TEAL,size=9)
    arrow(ax,(.34,.285),(.81,.285),"AGREEMENT delivered",color=TEAL)
    arrow(ax,(.81,.23),(.34,.23),"AGREEMENT delivered",color=TEAL)
    box(ax,.575,.14,"Matching vectors + latest local compatibility confirmed",w=.76,h=.065,color=TEAL)
    box(ax,.575,.04,"Independent final collision observer -> execution t=0, then t=1...5",w=.82,h=.065,color=BLUE,size=9)
    arrow(ax,(.575,.102),(.575,.078),color=BLUE)
    save(fig,"decentralized_negotiation_sequence")


if __name__ == "__main__":
    astar_workflow()
    centralized_workflow()
    architecture()
    negotiation_sequence()
    print(ROOT / "docs/diagrams")
