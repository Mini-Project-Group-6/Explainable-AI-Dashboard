# Ethics (IRB) materials — S4

| File | For | Read by the app? |
|---|---|---|
| `participant_information_sheet.md` | Participants, ethics committee | Yes, shown before consent |
| `consent_form.md` | Participants, ethics committee | Yes; the numbered statements are what "I agree" confirms |
| `survey_instruments.md` | Ethics committee, methods section | No. Generated from `app/study/instruments.py` |
| `data_management_plan.md` | Ethics committee | No |

The app shows the two participant documents exactly as they are written here.
Edit them here, never in code.

## Status: draft, not submitted

`python -m app.study status` lists everything outstanding. At the time of
writing, that is:

1. **Placeholders.** Every `{{LIKE_THIS}}` in the participant documents must
   be filled in. The dashboard treats the materials as unapproved while any
   remain.
2. **Item wording.** The `source_text` of each item in
   `app/study/instruments.py` was written for this draft. It was not copied
   from the papers. Check each scale word for word against its source, then
   add the source key to `_VERIFIED_SOURCES`.
3. **Claims the software cannot make true.** The documents promise the
   following. Confirm each with the colleges before submission:
   - declining does not affect marks or placement;
   - the dashboard's scores are practice feedback and not part of assessment;
   - tutors are not told who took part. The software shows tutors nothing,
     but a tutor who is also on the research team would still see the data.
4. **Electronic consent.** Consent is a recorded click with a timestamp, not a
   signature. Check that the committee accepts this.
5. **Data protection.** Check whether the study must register with Ghana's
   Data Protection Commission under Act 843.
6. **The ethics committee.** Name the committee and use its own forms. These
   files supply the content; the committee may require its own template.

## Recording approval

1. Freeze the materials. Fill in the placeholders, verify the sources, and run
   `python -m app.study instruments --write`.
2. Run `python -m app.study status` and note the **materials fingerprint**.
   Submit exactly these files.
3. Once approved, set both values in `app/study/approval.py`:

   ```python
   IRB_PROTOCOL = "<the committee's reference>"
   APPROVED_FINGERPRINT = "<the fingerprint from step 2>"
   ```

4. Run `status` again. It should report **Approved**.

Do not write the protocol number into the participant documents. That changes
the fingerprint the committee approved. The dashboard shows the number from
`IRB_PROTOCOL` instead.

**Any later change voids approval automatically.** This covers any wording
change to an item or a participant document. It also covers
`COTEACH_POST_AFTER_SUBMISSIONS` (the information sheet says "two more lesson
plans") and the set of participant roles. After such a change, new responses
are stored as pilot data until an amendment is approved and the new
fingerprint is recorded.

## Collecting and exporting

- Pilot data is anything collected before approval. It is kept apart
  automatically, and the export leaves it out unless `--include-pilot` is
  given.
- `python -m app.study export --out data/export` writes `participants.csv`,
  `responses.csv` and `submissions.csv` for S5. Researcher accounts can
  download the same files from the Trust survey tab. `data/` is gitignored, so
  exports never reach the repository. Keep them out of shared drives that are
  not covered by the data-management plan.
