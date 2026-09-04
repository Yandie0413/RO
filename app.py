"""
Ordonnancement des tâches — Méthode des Potentiels Métra (MPM)
================================================================
Application Streamlit : saisie dynamique des tâches (durée + antécédents),
calcul des dates au plus tôt / au plus tard / marges / chemin critique,
et affichage sous forme de tableau et de graphe interactif.
"""

import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network

st.set_page_config(page_title="Ordonnancement MPM", layout="wide")

# ----------------------------------------------------------------------------
# État initial : tableau vide, c'est à l'utilisateur de saisir ses tâches.
# ----------------------------------------------------------------------------
EMPTY_TASKS = pd.DataFrame(columns=["Tâche", "Durée", "Antécédents"])

if "tasks" not in st.session_state:
    st.session_state.tasks = EMPTY_TASKS.copy()


# ----------------------------------------------------------------------------
# Calcul MPM
# ----------------------------------------------------------------------------
def parse_preds(raw):
    if pd.isna(raw) or str(raw).strip() == "":
        return []
    return [p.strip() for p in str(raw).split(",") if p.strip()]


def compute_mpm(df: pd.DataFrame):
    df = df.copy()
    df["Tâche"] = df["Tâche"].astype(str).str.strip()
    df = df[df["Tâche"] != ""].reset_index(drop=True)

    if df.empty:
        raise ValueError("Ajoutez au moins une tâche.")

    if df["Tâche"].duplicated().any():
        dups = sorted(set(df.loc[df["Tâche"].duplicated(), "Tâche"]))
        raise ValueError(f"Identifiants de tâches en double : {', '.join(dups)}")

    if (df["Tâche"].str.lower().isin(["début", "debut", "fin"])).any():
        raise ValueError("« Début » et « Fin » sont réservés (tâches fictives), choisissez un autre nom.")

    df["Durée"] = pd.to_numeric(df["Durée"], errors="coerce")
    if df["Durée"].isna().any():
        bad = df.loc[df["Durée"].isna(), "Tâche"].tolist()
        raise ValueError(f"Durée invalide pour : {', '.join(bad)}")
    if (df["Durée"] < 0).any():
        raise ValueError("Les durées doivent être positives ou nulles.")

    df["preds"] = df["Antécédents"].apply(parse_preds)

    task_ids = set(df["Tâche"])
    for _, row in df.iterrows():
        for p in row["preds"]:
            if p not in task_ids:
                raise ValueError(f"La tâche « {row['Tâche']} » référence un antécédent inconnu : « {p} »")
        if row["Tâche"] in row["preds"]:
            raise ValueError(f"La tâche « {row['Tâche']} » ne peut pas être son propre antécédent.")

    durations = dict(zip(df["Tâche"], df["Durée"]))
    preds_map = dict(zip(df["Tâche"], df["preds"]))

    G = nx.DiGraph()
    G.add_node("Début")
    G.add_node("Fin")
    for t in task_ids:
        G.add_node(t)

    for t in df["Tâche"]:
        preds = preds_map[t]
        if not preds:
            G.add_edge("Début", t, weight=0)
        else:
            for p in preds:
                G.add_edge(p, t, weight=durations[p])

    for t in task_ids:
        if G.out_degree(t) == 0:
            G.add_edge(t, "Fin", weight=durations[t])

    if not nx.is_directed_acyclic_graph(G):
        cycle = nx.find_cycle(G)
        cycle_str = " → ".join(u for u, v in cycle) + f" → {cycle[-1][1]}"
        raise ValueError(f"Le graphe contient un cycle (planning impossible) : {cycle_str}")

    order = list(nx.topological_sort(G))

    # Dates au plus tôt (parcours avant)
    t_early = {n: 0 for n in G.nodes}
    for n in order:
        preds = list(G.predecessors(n))
        if preds:
            t_early[n] = max(t_early[p] + G[p][n]["weight"] for p in preds)

    project_duration = t_early["Fin"]

    # Dates au plus tard (parcours arrière)
    t_late = {n: project_duration for n in G.nodes}
    for n in reversed(order):
        succs = list(G.successors(n))
        if succs:
            t_late[n] = min(t_late[s] - G[n][s]["weight"] for s in succs)

    marge = {n: t_late[n] - t_early[n] for n in G.nodes}

    critical_edges = []
    for u, v, data in G.edges(data=True):
        w = data["weight"]
        if marge[u] == 0 and marge[v] == 0 and t_early[u] + w == t_early[v]:
            critical_edges.append((u, v))

    rows = []
    for t in df["Tâche"]:
        preds = preds_map[t]
        succs = [s if s != "Fin" else "fin" for s in G.successors(t)]
        rows.append(
            {
                "Tâche": t,
                "Durée": durations[t],
                "Antécédents": ", ".join(preds) if preds else "—",
                "Successeurs": ", ".join(succs) if succs else "—",
                "Date au plus tôt": t_early[t],
                "Date au plus tard": t_late[t],
                "Marge totale": marge[t],
                "Critique": "Oui" if marge[t] == 0 else "",
            }
        )
    result_df = pd.DataFrame(rows)

    return {
        "graph": G,
        "durations": durations,
        "t_early": t_early,
        "t_late": t_late,
        "marge": marge,
        "critical_edges": critical_edges,
        "project_duration": project_duration,
        "result_df": result_df,
    }


def get_critical_path(mpm):
    adj = {}
    for u, v in mpm["critical_edges"]:
        adj.setdefault(u, []).append(v)
    path = ["Début"]
    cur = "Début"
    seen = {cur}
    while cur != "Fin":
        nxts = adj.get(cur, [])
        if not nxts:
            break
        cur = nxts[0]
        if cur in seen:
            break
        seen.add(cur)
        path.append(cur)
    return [p for p in path if p not in ("Début", "Fin")]


# ----------------------------------------------------------------------------
# Rendu du graphe MPM (pyvis)
# ----------------------------------------------------------------------------
def render_mpm_graph(mpm):
    G = mpm["graph"]
    t_early, t_late, marge, durations = mpm["t_early"], mpm["t_late"], mpm["marge"], mpm["durations"]
    critical_edges = set(mpm["critical_edges"])

    net = Network(height="620px", width="100%", directed=True, bgcolor="#ffffff", cdn_resources="in_line")
    net.set_options(
        """
        {
          "layout": {
            "hierarchical": {
              "enabled": true,
              "direction": "LR",
              "sortMethod": "directed",
              "nodeSpacing": 160,
              "levelSeparation": 200
            }
          },
          "physics": { "enabled": false },
          "interaction": { "hover": true, "zoomView": true, "dragView": true },
          "edges": {
            "arrows": { "to": { "enabled": true, "scaleFactor": 0.7 } },
            "font": { "align": "top", "size": 14 },
            "smooth": { "type": "cubicBezier", "roundness": 0.4 }
          },
          "nodes": { "font": { "size": 14, "multi": false } }
        }
        """
    )

    for n in G.nodes:
        if n in ("Début", "Fin"):
            label = f"{n}\n{t_early[n]:g}"
            net.add_node(
                n, label=label, shape="circle", size=30,
                color={"background": "#1f77b4", "border": "#144d79"},
                font={"color": "#ffffff"},
                borderWidth=2,
            )
        else:
            crit = marge[n] == 0
            # Format proche du cours : cercle avec tôt | tard, nom en légende,
            # durée/marge disponibles au survol (comme les annotations du cours).
            label = f"{n}\n{t_early[n]:g} | {t_late[n]:g}"
            title = f"{n} — durée {durations[n]:g} — marge {marge[n]:g}"
            color = (
                {"background": "#fde2e1", "border": "#d62728"}
                if crit
                else {"background": "#ffffff", "border": "#333333"}
            )
            net.add_node(
                n, label=label, title=title, shape="circle", size=32,
                color=color, borderWidth=3 if crit else 1,
                font={"size": 13},
            )

    for u, v, data in G.edges(data=True):
        crit = (u, v) in critical_edges
        net.add_edge(
            u, v,
            label=f"{data['weight']:g}",
            color="#d62728" if crit else "#999999",
            width=3 if crit else 1,
        )

    html = net.generate_html(notebook=False)
    st.html(html, unsafe_allow_javascript=True)


# ----------------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------------
st.title("Ordonnancement des tâches — Graphe MPM")
st.caption(
    "Méthode des Potentiels Métra : saisissez vos tâches, les dates au plus tôt / au plus tard, "
    "les marges et le chemin critique se recalculent automatiquement."
)

col_left, col_right = st.columns([3, 1])
with col_right:
    if st.button("Vider le tableau"):
        st.session_state.tasks = EMPTY_TASKS.copy()
        st.rerun()

st.subheader("1. Saisie des tâches")
st.session_state.tasks = st.data_editor(
    st.session_state.tasks,
    num_rows="dynamic",
    key="task_editor",
    column_config={
        "Tâche": st.column_config.TextColumn("Tâche", required=True, help="Identifiant unique, ex: A"),
        "Durée": st.column_config.NumberColumn("Durée", required=True, min_value=0, step=1),
        "Antécédents": st.column_config.TextColumn(
            "Antécédents", help="Identifiants séparés par des virgules, ex: A,B (laisser vide si aucun)"
        ),
    },
)

st.divider()

try:
    mpm = compute_mpm(st.session_state.tasks)
except ValueError as e:
    st.error(str(e))
    st.stop()

m1, m2, m3 = st.columns(3)
m1.metric("Durée totale du projet", f"{mpm['project_duration']:g}")
crit_path = get_critical_path(mpm)
m2.metric("Nombre de tâches critiques", len(crit_path))
m3.metric("Chemin critique", " → ".join(crit_path) if crit_path else "—")

st.subheader("2. Tableau des tâches, dates, marges et successeurs")
st.caption("Comme dans le cours : Tâche, Durée, Antécédents, Successeurs, dates au plus tôt/tard, marge totale.")


def highlight_critical(row):
    return ["background-color: #fde2e1" if row["Critique"] == "Oui" else "" for _ in row]


st.dataframe(
    mpm["result_df"].style.apply(highlight_critical, axis=1),
    hide_index=True,
)
st.download_button(
    "Exporter le tableau (CSV)",
    mpm["result_df"].to_csv(index=False).encode("utf-8-sig"),
    file_name="ordonnancement_mpm.csv",
    mime="text/csv",
)

st.subheader("3. Graphe MPM (nœuds = tâches, arcs = contraintes d'antériorité)")
st.caption(
    "Chaque cercle affiche tôt | tard (survolez pour la durée et la marge). "
    "En rouge : tâches et chemin critique (marge nulle). Glissez / zoomez pour explorer."
)
render_mpm_graph(mpm)
