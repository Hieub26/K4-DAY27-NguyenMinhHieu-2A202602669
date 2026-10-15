"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

Goal: wrap the LangGraph built in exercises 1–4 in a web UI that adapts to
the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    > 72%        auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer
"""

from __future__ import annotations

import asyncio
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_conn, db_path
from exercises.exercise_4_audit import build_graph


load_dotenv()


# ─── Session state ─────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pr_url" not in st.session_state:
    st.session_state.pr_url = ""
if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None
if "final" not in st.session_state:
    st.session_state.final = None


# ─── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="HITL PR Review", layout="wide")
st.title("HITL PR Review Agent")


# ─── Data helpers ──────────────────────────────────────────────────────────
async def fetch_recent_threads(limit: int = 15) -> list[dict]:
    """Same query as `audit.replay.list_threads`, but returns rows instead of printing."""
    async with db_conn() as conn:
        async with conn.execute(
            """
            SELECT thread_id,
                   pr_url,
                   MAX(timestamp)  AS last_event,
                   MAX(CASE risk_level WHEN 'high' THEN 3 WHEN 'med' THEN 2 ELSE 1 END) AS worst,
                   COUNT(*)        AS events
              FROM audit_events
             GROUP BY thread_id, pr_url
             ORDER BY MAX(timestamp) DESC
             LIMIT ?
            """,
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def load_thread(thread_id: str) -> dict:
    """Read the checkpointed graph state of a thread (pending interrupt or final values)."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        snap = await app.aget_state({"configurable": {"thread_id": thread_id}})
    interrupts = [i for t in snap.tasks for i in t.interrupts]
    return {
        "values": snap.values,
        "interrupt": interrupts[0].value if interrupts else None,
    }


def open_thread(thread_id: str) -> None:
    """Jump back into an existing session (resume a pending interrupt or show its result)."""
    loaded = asyncio.run(load_thread(thread_id))
    st.session_state.thread_id = thread_id
    st.session_state.pr_url = loaded["values"].get("pr_url", "")
    st.session_state.interrupt_payload = loaded["interrupt"]
    st.session_state.final = None if loaded["interrupt"] else loaded["values"]


# ─── Sidebar — recent sessions ─────────────────────────────────────────────
RISK_BADGE = {3: "🔴 high", 2: "🟡 med", 1: "🟢 low"}

with st.sidebar:
    st.header("Recent sessions")
    try:
        threads = asyncio.run(fetch_recent_threads())
    except Exception as e:  # e.g. DB locked / unreadable
        threads = []
        st.caption(f"Could not read audit_events: {e}")
    if not threads:
        st.caption("No sessions yet — run a review.")
    for t in threads:
        pr_short = t["pr_url"].split("github.com/")[-1]
        label = f"{pr_short} · {RISK_BADGE.get(t['worst'], '?')} · {t['events']} ev"
        current = t["thread_id"] == st.session_state.thread_id
        if st.button(label, key=f"thread_{t['thread_id']}", type="primary" if current else "secondary",
                     help=f"{t['thread_id']}\nlast event {t['last_event']}", width="stretch"):
            open_thread(t["thread_id"])
            st.rerun()


# ─── Top form — start a new review ─────────────────────────────────────────
with st.form("start"):
    pr_url = st.text_input(
        "PR URL", value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("Run review")


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket: show the LLM review + 3 buttons. Return resume dict or None."""
    conf = payload["confidence"]
    st.subheader(f"Approval requested — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])

    for c in payload.get("comments", []):
        st.markdown(f"- **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}")

    with st.expander("Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")

    feedback = st.text_input("Feedback (optional)", key="approval_feedback")
    col1, col2, col3 = st.columns(3)
    if col1.button("Approve", type="primary", width="stretch"):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("Reject", width="stretch"):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("Edit", width="stretch"):
        if not feedback.strip():
            st.warning("Write what should change in the feedback box before choosing Edit.")
            return None
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    conf = payload["confidence"]
    st.subheader(f"Strong escalation — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.error("**Risk factors**\n\n" + "\n".join(f"- {r}" for r in payload["risk_factors"]))
    st.markdown(payload["summary"])

    with st.form("escalation"):
        answers: dict[str, str] = {}
        for i, q in enumerate(payload["questions"]):
            answers[q] = st.text_area(q, key=f"esc_q_{i}", height=80)
        if st.form_submit_button("Submit answers", type="primary"):
            if any(not a.strip() for a in answers.values()):
                st.warning("Please answer every question.")
                return None
            return {q: a.strip() for q, a in answers.items()}
    return None


# ─── Drive the graph ───────────────────────────────────────────────────────
async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Invoke the graph once. Returns the final result or {'__interrupt__': ...}."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}
        if resume_value is None:
            return await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        return await app.ainvoke(Command(resume=resume_value), cfg)


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted and pr_url:
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None

    try:
        with st.spinner("Fetching PR + asking the LLM..."):
            result = asyncio.run(run_graph(pr_url, st.session_state.thread_id))
    except Exception as e:
        st.error(f"Review failed: {e}")
        st.stop()

    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
    else:
        st.session_state.final = result

if st.session_state.thread_id:
    st.caption(f"thread_id = `{st.session_state.thread_id}`")

# Render the current interrupt card, if any
payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload["kind"]
    answer = render_approval_card(payload) if kind == "approval_request" else render_escalation_card(payload)
    if answer is not None:
        with st.spinner("Resuming..."):
            result = asyncio.run(run_graph(
                st.session_state.pr_url, st.session_state.thread_id, resume_value=answer,
            ))
        if "__interrupt__" in result:
            st.session_state.interrupt_payload = result["__interrupt__"][0].value
        else:
            st.session_state.interrupt_payload = None
            st.session_state.final = result
        st.rerun()

# Render final state, if reached
if st.session_state.final is not None:
    final = st.session_state.final
    action = final.get("final_action") or "?"
    analysis = final.get("analysis")
    pr = st.session_state.pr_url

    if action == "commit_failed" or action.endswith("commit_failed"):
        st.error("Posting the comment to GitHub failed — check GITHUB_TOKEN scope.")
    elif action.startswith("auto"):
        st.success(f"✓ Auto-approved — agent posted the review itself ({action})")
    elif action.startswith("committed"):
        if final.get("escalation_answers"):
            st.success("✓ Refined review posted after escalation")
        else:
            st.success("✓ Approved — review comment posted")
    elif action == "rejected":
        st.warning(f"Not posted (reviewer chose `{final.get('human_choice')}`)")
    else:
        st.info(f"final_action = {action}")

    if analysis is not None:
        st.metric("Final confidence", f"{analysis.confidence:.0%}")
        st.markdown(analysis.summary)
        for c in analysis.comments:
            st.markdown(f"- **[{c.severity}]** `{c.file}:{c.line or '?'}` — {c.body}")
    if final.get("escalation_answers"):
        with st.expander("Reviewer answers"):
            for q, a in final["escalation_answers"].items():
                st.markdown(f"**{q}**\n\n{a}")
    if pr:
        st.link_button("View comment on GitHub", pr)
    st.caption(f"replay: `uv run python -m audit.replay --thread {st.session_state.thread_id}`")
