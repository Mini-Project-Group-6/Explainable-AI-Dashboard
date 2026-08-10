# app/session.py
"""Sign-in state and the page gate. S3 owns auth.

Three problems in the first draft, all fixed:

  * the sign-in form shipped with a working password prefilled into the widget;
  * ``pages/dashboard.py`` rendered for anyone who typed its URL;
  * credentials were never actually checked. Unless ``APP_EMAIL``/
    ``APP_PASSWORD`` were set — which by default they were not — any email with
    any non-empty password was admitted, silently.

Passwords are now verified against hashed accounts in the database (see
``app.accounts``). With no accounts, nobody gets in: for a study collecting
identifiable data from student teachers, "not configured yet" has to mean
locked, not open.

Two protections beyond the password check, because this runs on shared college
lab machines:

  * **Lockout.** Repeated failures against one address stop being answered for
    a while. Held in process memory, which is the right scope — it is shared by
    every browser session hitting this server, so opening a new tab does not
    reset it.
  * **Idle timeout.** A session left open on a lab machine expires, so the next
    person at that desk does not inherit it.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import streamlit as st

from app import accounts

_USER_KEY = "auth_user"
_SEEN_KEY = "auth_last_seen"

#: Sign-in attempts allowed per address before it is locked out.
MAX_ATTEMPTS = 5
LOCKOUT_SECONDS = 15 * 60

#: Idle time before a session is dropped. Short by default: these are shared
#: machines in a computer lab, not personal laptops.
IDLE_TIMEOUT_SECONDS = int(os.getenv("APP_IDLE_TIMEOUT", 60 * 60))

#: address -> (consecutive failures, time of last failure). Process-wide on
#: purpose; per-session state would reset with every new tab.
_failures: dict[str, tuple[int, float]] = {}


@dataclass(frozen=True)
class User:
    email: str
    name: str
    role: str = "student_teacher"

    @property
    def initials(self) -> str:
        parts = [p for p in self.name.replace(".", " ").split() if p]
        return "".join(p[0] for p in parts[:2]).upper() or self.email[:2].upper()

    @property
    def role_label(self) -> str:
        return self.role.replace("_", " ").capitalize()


# ---------------------------------------------------------------------------
# Lockout
# ---------------------------------------------------------------------------

def _lockout_remaining(email: str) -> int:
    """Seconds left on a lockout for *email*, or 0."""
    attempts, last = _failures.get(email, (0, 0.0))
    if attempts < MAX_ATTEMPTS:
        return 0
    remaining = int(LOCKOUT_SECONDS - (time.time() - last))
    if remaining <= 0:
        _failures.pop(email, None)
        return 0
    return remaining


def _record_failure(email: str) -> None:
    attempts, _ = _failures.get(email, (0, 0.0))
    _failures[email] = (attempts + 1, time.time())


# ---------------------------------------------------------------------------
# Configuration state
# ---------------------------------------------------------------------------

def accounts_exist() -> bool:
    """False when the deployment has no accounts and nobody can sign in."""
    try:
        return accounts.count_accounts() > 0
    except Exception:                                 # noqa: BLE001
        return False


def credentials_enforced() -> bool:
    """True when a password actually has to be correct to get in.

    Now always true. Kept because the transparency panel reports it, and
    because a future SSO path should be able to answer the same question.
    """
    return True


# ---------------------------------------------------------------------------
# Sign in / out
# ---------------------------------------------------------------------------

def sign_in(email: str, password: str) -> tuple[bool, str]:
    """Validate credentials and start a session. Returns (ok, message)."""
    email = accounts.normalise_email(email)
    if not email or "@" not in email:
        return False, "Enter a valid email address."
    if not password:
        return False, "Enter your password."

    if not accounts_exist():
        return False, ("This deployment has no accounts yet, so sign-in is "
                       "closed. An administrator must create one first.")

    locked = _lockout_remaining(email)
    if locked:
        return False, (f"Too many failed attempts. Try again in "
                       f"{locked // 60 + 1} minute(s).")

    account = accounts.verify_credentials(email, password)
    if account is None:
        _record_failure(email)
        # One message for both causes: revealing "no such account" would let
        # anyone enumerate which student teachers are enrolled.
        return False, "Email or password is incorrect."

    _failures.pop(email, None)
    st.session_state[_USER_KEY] = {
        "email": account.email, "name": account.name, "role": account.role}
    _touch()
    return True, ""


def sign_out() -> None:
    st.session_state.pop(_USER_KEY, None)
    st.session_state.pop(_SEEN_KEY, None)
    # Scored submissions belong to the signed-in user; leaving them in session
    # state would show the next person to sign in someone else's lesson plan.
    for key in ("submission", "session_history", "scored_submissions",
                "recorded_submissions", "shap_criterion"):
        st.session_state.pop(key, None)


def _touch() -> None:
    st.session_state[_SEEN_KEY] = time.time()


def _expired() -> bool:
    last_seen = st.session_state.get(_SEEN_KEY)
    if last_seen is None:
        return False
    return (time.time() - last_seen) > IDLE_TIMEOUT_SECONDS


def current_user() -> Optional[User]:
    """The signed-in user, or None. Expires an idle session as a side effect."""
    stored = st.session_state.get(_USER_KEY)
    if not stored:
        return None
    if _expired():
        sign_out()
        st.session_state["auth_expired"] = True
        return None
    _touch()
    return User(email=stored["email"], name=stored["name"],
                role=stored.get("role", "student_teacher"))


def require_user() -> User:
    """Gate an inner page. Sends unauthenticated visitors back to sign-in.

    Typing the dashboard URL directly used to render the whole page.
    """
    user = current_user()
    if user is not None:
        return user
    try:
        st.switch_page("main.py")
    except Exception:                                 # noqa: BLE001
        st.warning("Please sign in to open the dashboard.")
    st.stop()
