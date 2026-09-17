# Data management plan

**Study:** {{STUDY_TITLE}}

**Data controller:** {{DATA_CONTROLLER}}

This plan describes the data the co-teaching dashboard holds, who can reach it,
and what happens to it. It describes the software as built. Section 8 lists the
arrangements the research team still has to confirm.

## 1. Participants and recruitment

Participants are student teachers at {{PARTICIPATING_COLLEGES}} with a
dashboard account. Accounts are created by the research team. There is no
self-registration.

The first time a student teacher signs in, the dashboard shows the
participant information sheet and consent form, with two equally weighted
choices: agree, or decline. Declining has no effect on use of the dashboard,
and the question is not asked again. Tutor and researcher accounts are never
invited.

## 2. Data collected

| Data | Collected from | Purpose |
|---|---|---|
| Name, institutional email, role | Account creation (all users) | Sign-in |
| Password | Account creation | Sign-in; stored only as a salted scrypt hash |
| Lesson plan file | Upload (all users) | Scoring. Written to a private temporary file and deleted as soon as it has been scored |
| File name, upload time, overall score, band, ten criterion scores, model version | Upload (all users) | The student's revision history; for participants, the plan-quality outcome |
| Consent decision, time, random participant code, version of the materials agreed to, withdrawal time | Consent screen (student teachers) | Proof of consent; withdrawal |
| Survey answers (22 statements in the first survey and 30 in the follow-up, on a 1–5 scale; two optional background questions in the first), completion time, number of plans scored before the survey, instrument version | Surveys (participants only) | The trust and acceptance measures |

No other personal data is collected. The two background questions (level of
study and previous use of AI tools) both offer "Prefer not to say".

## 3. Pseudonymisation

On agreeing, each participant is given a random code of the form
`P-` followed by eight hexadecimal digits. The code is generated at random, so
it cannot be worked out from the person's identity.

The research dataset is produced by the dashboard's export
(`python -m app.study export`). It contains participant codes only. It never
contains names, email addresses or file names — file names are excluded
because student teachers often name files after themselves. Declined and
withdrawn participants, and all tutor and researcher accounts, are left out
entirely.

The link between code and email exists only inside the dashboard database. It
is kept so that a participant can withdraw.

## 4. Storage and security

| Measure | Detail |
|---|---|
| Location | {{STORAGE_LOCATION}} |
| Database | PostgreSQL, accessed by the dashboard through its own login role, which owns only the study database |
| Access to the dashboard | Password sign-in; no anonymous access. With no accounts configured, nobody can sign in |
| Passwords | scrypt (n = 2¹⁴, r = 8, p = 1) with a per-account random salt; at least 10 characters |
| Guessing | After 5 failed sign-ins, an address is locked for 15 minutes. Unknown and known addresses give the same error message and take the same time |
| Shared lab machines | Idle sessions end after 60 minutes. Signing out clears the session, including any partly answered survey |
| Separation of roles | Student teachers see only their own submissions. Tutors see no study data. Only researcher accounts can see enrolment counts or download the export |
| Exported files | Stored on {{EXPORT_STORAGE}}, available only to the research team |
| Backups | A verified `pg_dump` of the database every day at 18:00, to {{BACKUP_LOCATION}}. The 14 most recent are kept; older ones are deleted automatically. |

## 5. Withdrawal

A participant can withdraw at any time from the dashboard's Trust survey tab.
Withdrawal:

- deletes the participant's survey answers from the database immediately;
- marks the consent record as withdrawn, so the person is not invited again;
- excludes the participant's scored plans from every later export. The plans
  stay in the participant's own revision history, because that history is part
  of the dashboard's normal service.

Exports taken before a withdrawal are not changed automatically. Before
analysis, the research team re-exports, or removes the withdrawn code from
earlier exports. Once group results have been analysed, after
{{WITHDRAWAL_DEADLINE}}, those results cannot be changed.

## 6. Pilot data

Until the committee's approval is recorded in the software, every survey
response is stored without a protocol reference and marked as pilot data. The
export leaves pilot data out unless the team explicitly asks for it. After
approval, a response is tied to the approved materials by a fingerprint of the
exact wording. If any wording is changed afterwards, new responses revert to
pilot status until the change is approved.

## 7. Retention and disposal

Study data is kept for {{RETENTION_PERIOD}} after the end of the study and then
deleted from the database and the exports. Backups roll over every 14 days, so
anything deleted from the database, including a withdrawn participant's
answers, is gone from every backup within 14 days. Published results
contain only aggregate figures.

Data is processed in line with Ghana's Data Protection Act, 2012 (Act 843).

## 8. To be confirmed by the research team

- Where the server and exports are held, and who administers them.
- Where backups are kept. The daily dump is written on the server itself,
  which protects against mistakes but not against losing the machine; a copy
  somewhere else needs its own arrangement, and the same 14-day rule.
- The retention period and the withdrawal deadline.
- Who holds researcher accounts.
