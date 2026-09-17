# app/study/ui.py
"""Streamlit widgets for the study. The dashboard calls these and nothing else.

    st_consent_card     the one-time participation choice, above the tabs
    st_upload_notice    why the uploader is disabled, when it is
    st_survey_tab       the whole "Trust survey" tab, for every role

Every widget key starts with ``study_`` so ``session.sign_out`` can clear the
lot; a half-answered survey must not be waiting for the next person at a
shared lab machine.
"""

from __future__ import annotations

import time

import streamlit as st

from app.study import approval, documents, export, flow, instruments, store

KEY_PREFIX = "study_"


def _status_caption() -> None:
    if approval.is_approved():
        st.caption(f"Approved by the ethics committee · protocol "
                   f"{approval.IRB_PROTOCOL}")
    else:
        st.caption(":orange[Pilot version — these study materials have not "
                   "yet been approved by the ethics committee, and answers "
                   "given now are kept apart from study data.]")


def _consent_form(user, *, allow_decline: bool, key: str) -> None:
    statements = documents.consent_statements()
    if not statements:
        st.error("The consent form could not be loaded, so participation "
                 "cannot be recorded. Please tell the research team.",
                 icon=":material/error:")
        return

    with st.expander("Participant information sheet"):
        st.markdown(documents.body(documents.INFORMATION_SHEET))

    st.markdown("**By choosing “I agree to take part”, you confirm that:**")
    st.markdown("\n".join(f"{n}. {text}" for n, text in enumerate(statements, 1)))

    # Same weight for both buttons: a highlighted "agree" is a nudge, and
    # participation has to be a free choice.
    left, right = st.columns(2)
    agree = left.button("I agree to take part", key=f"{key}agree",
                        icon=":material/check:", width="stretch")
    decline = allow_decline and right.button(
        "No thanks — just use the dashboard", key=f"{key}decline",
        icon=":material/close:", width="stretch")

    if agree or decline:
        if store.record_consent(user.email, agreed=bool(agree)) is None:
            st.error("Your choice could not be saved. Please try again.",
                     icon=":material/error:")
            return
        st.rerun()


def st_consent_card(user) -> None:
    with st.container(border=True):
        st.markdown("#### Research participation")
        st.markdown(
            "This dashboard is part of a study on whether explanations of AI "
            "feedback help student teachers trust that feedback and improve "
            "their lesson plans. Taking part means answering two short "
            "surveys, about 10 minutes each. **Saying no does not affect your "
            "use of the dashboard in any way**, and you will not be asked "
            "again.")
        _status_caption()
        _consent_form(user, allow_decline=True, key=f"{KEY_PREFIX}card_")


def st_upload_notice(state: flow.StudyState) -> None:
    if state.upload_blocked:
        st.info(state.block_reason, icon=":material/assignment:")


# ---------------------------------------------------------------------------
# Survey form
# ---------------------------------------------------------------------------

def _radio_key(wave: str, item_id: str) -> str:
    return f"{KEY_PREFIX}{wave}_{item_id}"


def _survey_form(user, wave: str) -> None:
    started_key = f"{KEY_PREFIX}started_{wave}"
    started = st.session_state.setdefault(started_key, time.time())
    items = instruments.items_for(wave)
    background = instruments.background_for(wave)

    if wave == "pre":
        st.markdown("#### First survey")
        st.markdown("Please answer before your first lesson plan is scored. "
                    "There are no right or wrong answers — we want your own "
                    "view of the dashboard as you expect it to be.")
    else:
        st.markdown("#### Follow-up survey")
        st.markdown("Now that you have used the dashboard, please answer "
                    "with your experience of it in mind. Some statements "
                    "repeat the first survey; that is deliberate.")
    st.caption(f"{len(items)} statements · every statement needs an answer"
               + (" · the background questions are optional" if background else ""))

    with st.form(f"{KEY_PREFIX}form_{wave}", border=False):
        number = 0
        for heading, section in instruments.sections_for(wave):
            st.markdown(f"##### {heading}")
            for item in section:
                number += 1
                # No default: a preselected answer is an answer nobody gave.
                st.radio(f"**{number}.** {item.text}", instruments.LIKERT,
                         index=None, horizontal=True,
                         format_func=instruments.LIKERT_LABELS.get,
                         key=_radio_key(wave, item.id))
        if background:
            st.markdown("##### About you")
            for question in background:
                st.radio(question.text, question.options, index=None,
                         horizontal=True, key=_radio_key(wave, question.id))
        submitted = st.form_submit_button("Submit survey", type="primary",
                                          icon=":material/send:")

    if not submitted:
        return
    answers = {entry.id: st.session_state.get(_radio_key(wave, entry.id))
               for entry in (*items, *background)}
    saved, problems = store.record_response(
        user, wave, answers, duration_seconds=time.time() - started)
    if not saved:
        st.error(" ".join(problems), icon=":material/error:")
        return
    for key in [k for k in st.session_state if str(k).startswith(
            (f"{KEY_PREFIX}{wave}_", started_key))]:
        del st.session_state[key]
    st.session_state[f"{KEY_PREFIX}thanks"] = wave
    st.rerun()


def _withdraw_panel(user, state: flow.StudyState) -> None:
    with st.expander("Withdraw from the study"):
        st.markdown(
            "You can stop taking part at any time without giving a reason. "
            "Withdrawing **deletes your survey answers straight away** and "
            "leaves your scored plans out of the research data. You can keep "
            "using the dashboard as normal.")
        if state.participant_code:
            st.markdown(f"Your participant code is `{state.participant_code}`. "
                        "Quote it if you contact the research team.")
        sure = st.checkbox("I want to withdraw from the study",
                           key=f"{KEY_PREFIX}withdraw_confirm")
        if st.button("Withdraw", disabled=not sure,
                     key=f"{KEY_PREFIX}withdraw", icon=":material/logout:"):
            if store.withdraw(user.email):
                st.rerun()
            st.error("Your withdrawal could not be saved. Please try again.",
                     icon=":material/error:")


# ---------------------------------------------------------------------------
# Researcher view
# ---------------------------------------------------------------------------

def _researcher_panel() -> None:
    st.markdown("#### Study administration")
    problems = approval.approval_problems()
    if problems:
        st.warning("**Draft materials — not approved for data collection.**\n\n"
                   + "\n".join(f"- {p}" for p in problems),
                   icon=":material/gavel:")
    else:
        st.success(f"Approved · protocol {approval.IRB_PROTOCOL}",
                   icon=":material/verified:")
    st.caption(f"Instrument version {instruments.INSTRUMENT_VERSION} · "
               f"materials fingerprint `{approval.fingerprint()}`")

    counts = store.enrolment_counts()
    columns = st.columns(5)
    for column, (label, key) in zip(columns, (
            ("Taking part", "agreed"), ("Declined", "declined"),
            ("Withdrawn", "withdrawn"), ("First surveys", "pre"),
            ("Follow-ups", "post"))):
        column.metric(label, counts[key])

    st.markdown("##### Export for analysis")
    st.caption("Participant codes only — no emails, names or file names.")
    include_pilot = st.toggle(
        "Include pilot data (collected before approval)",
        value=not approval.is_approved(), key=f"{KEY_PREFIX}include_pilot")
    tables = export.build(include_pilot=include_pilot)
    columns = st.columns(len(tables))
    for column, (name, table) in zip(columns, tables.items()):
        column.download_button(
            f"{name}.csv ({len(table[1])} rows)",
            data=export.to_csv(table).encode("utf-8-sig"),
            file_name=f"{name}.csv", mime="text/csv",
            key=f"{KEY_PREFIX}download_{name}", width="stretch",
            icon=":material/download:")


# ---------------------------------------------------------------------------
# The tab
# ---------------------------------------------------------------------------

def st_survey_tab(user) -> None:
    if user.role == "researcher":
        _researcher_panel()
        return

    state = store.current_state(user)
    stage = state.stage

    if stage == flow.NOT_PARTICIPANT:
        st.info("The trust survey is for student teachers taking part in the "
                "study. Nothing is collected from this account.",
                icon=":material/info:")
        return
    if stage == flow.UNAVAILABLE:
        st.warning("Storage is unavailable, so survey answers cannot be "
                   "saved right now. The dashboard works as normal.",
                   icon=":material/warning:")
        return
    if stage == flow.UNDECIDED:
        st.info("Please choose whether to take part using the panel above "
                "the tabs.", icon=":material/assignment:")
        return
    if stage == flow.DECLINED:
        st.info("You chose not to take part. Nothing from your use of the "
                "dashboard is used in the study.", icon=":material/info:")
        with st.expander("Changed your mind?"):
            _consent_form(user, allow_decline=False, key=f"{KEY_PREFIX}late_")
        return
    if stage == flow.WITHDRAWN:
        st.info("You have withdrawn from the study and your survey answers "
                "have been deleted. You can keep using the dashboard. To "
                "rejoin, contact the research team.",
                icon=":material/info:")
        return

    thanks = st.session_state.pop(f"{KEY_PREFIX}thanks", None)
    if thanks:
        st.success("Thank you — your answers are saved."
                   + (" You can now upload a lesson plan in the Scoring tab."
                      if thanks == "pre" else ""),
                   icon=":material/check_circle:")

    _status_caption()
    if state.due_wave:
        _survey_form(user, state.due_wave)
    elif stage == flow.WAITING:
        done = state.plans_since_pre
        needed = flow.POST_AFTER_SUBMISSIONS
        st.markdown("#### Follow-up survey")
        st.markdown(
            f"The follow-up survey opens after you have had {needed} lesson "
            f"plans scored. You have had **{done}** scored since the first "
            f"survey — {state.plans_until_post} to go. Revising a plan and "
            f"uploading it again counts.")
        st.progress(done / needed if needed else 1.0)
    elif stage == flow.COMPLETE:
        st.markdown("#### All done")
        st.markdown("You have answered both surveys. Thank you for taking "
                    "part — keep using the dashboard as much as you like.")

    st.write("")
    _withdraw_panel(user, state)
