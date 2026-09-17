"""S4: the trust and acceptance study — instruments, consent, data collection.

    instruments   the survey items, their published sources, and scoring
    documents     the participant documents in docs/irb/, read at runtime
    approval      whether the current materials are the approved ones
    flow          where a person is in the study (pure logic)
    store         consent and response reads/writes
    export        anonymised CSVs for S5
    ui            the Streamlit widgets the dashboard calls

    python -m app.study status        approval state, fingerprint, enrolment
    python -m app.study instruments   print (or --write) the instrument appendix
    python -m app.study export --out data/export [--include-pilot]
"""
