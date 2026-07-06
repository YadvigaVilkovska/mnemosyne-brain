# Project: Dialogue Signal and Memory Extraction Pipeline

## 1. Purpose

This project defines a minimal, traceable pipeline for extracting useful information from dialogue without turning every user message into a memory update.

The system separates three responsibilities:

1. Detect surface-level signals in the current user message.
2. Analyze meaning across a dialogue segment.
3. Decide whether anything should affect durable memory.

The pipeline must be conservative, debuggable, and resistant to accidental memory writes.

## 2. Core architecture

```text
User message
  ↓
Stage 0: Current Signal Extraction
  ↓
Raw turn + current_signal saved
  ↓
Segment Analysis
  ↓
Memory Decision
  ↓
Durable memory write, skip, update, conflict, or ask
```

If existing code already uses the name `Stage 2`, the implementation may keep that historical name for compatibility.

```text
Stage 2 keeps its historical name for compatibility.
There is no runtime Stage 1 in this architecture.
```

## 3. Core principle

```text
Stage 0 is a sensor.
Segment Analysis is an interpreter.
Memory Decision is the durable-memory gate.
```

Stage 0 captures what is visible in the current message.

Segment Analysis interprets what a dialogue segment appears to mean.

Memory Decision compares candidate information with durable memory and decides what happens.

There is no separate Stage 1.

## 4. Four boundary laws

```text
Entity ≠ signal.
Signal ≠ fact.
Candidate fact ≠ memory candidate.
Memory candidate ≠ memory write.
```

These boundaries are mandatory.

Stage 0 may extract entities and signals.

Segment Analysis may produce candidate facts and candidate memory items.

Only Memory Decision may decide whether durable memory is changed.

---

# Stage 0: Current Signal Extraction

## 5. Responsibility

Stage 0 analyzes only the current user message.

Its job is to extract:

1. Entity mentions.
2. Information-bearing signals.
3. Unresolved references.
4. Ambiguous references.
5. Explicit interaction reaction signals, if visible in the message.

Stage 0 does not answer the user.

Stage 0 does not decide whether information is new.

Stage 0 does not decide whether information is already known.

Stage 0 does not decide whether information is memory-worthy.

Stage 0 does not create memory candidates.

Stage 0 does not write memory.

Stage 0 does not extract durable relations.

Stage 0 does not compare against durable memory.

Stage 0 does not infer stable user traits from one message.

## 6. Entity vs information signal

Entity means what is mentioned or referred to.

Information signal means what the current message appears to say about one or more mentioned entities.

Examples:

```text
A person name is an entity.

A role phrase is an entity-like mention.

A statement that the person has that role is an information signal.

A work activity may be mentioned as an entity-like concept.

A claim that someone performed that work activity is an information signal.
```

Stage 0 extracts both, but it must not convert them into durable facts.

## 7. Stage 0 input

```json
{
  "current_user_message": "",
  "recent_messages": [],
  "previous_stage0": null
}
```

`current_user_message` is the primary source.

`recent_messages` may only be used to resolve pronouns, aliases, vague references, or context-dependent wording inside `current_user_message`.

`recent_messages` must not create new information signals by itself.

## 8. Stage 0 output

```json
{
  "entities": [],
  "information_signals": [],
  "unresolved_references": [],
  "ambiguous_references": []
}
```

Stage 0 output must not contain:

```text
memory_candidates
selected_memory_ids
draft_answer
final_answer
```

## 9. Entity object

```json
{
  "id": "e1",
  "mention": "",
  "entity_type": "",
  "source_span": "",
  "resolution_status": "literal | resolved | unresolved | ambiguous",
  "resolved_to": null
}
```

## 10. Information signal object

```json
{
  "id": "s1",
  "source_span": "",
  "signal_type": "",
  "about_entity_ids": [],
  "signal_scope": "",
  "polarity": "asserted | negated | questioned | hypothetical | corrected | uncertain",
  "epistemic_status": "user_claim | user_question | user_correction | quoted_or_reported | instruction | user_reaction",
  "extraction_note": null
}
```

`extraction_note` may only explain why the fragment was extracted as a signal.

It must not contain deeper interpretation, memory judgment, durable relation analysis, or personality inference.

## 11. Allowed entity types

```text
person_reference
name_mention
alias_mention
pronoun
relationship_role
occupational_role
work_activity
organization
project
task
document
place
address
account
communication_identifier
site
event
time_expression
money_expression
financial_object
object
topic
sensitive_context
unknown_reference
```

`interaction_target` is intentionally not an entity type.

Current-interaction objects should be handled through `signal_scope: about_current_interaction`.

## 12. Allowed signal types

```text
identity
name_or_alias
role
occupation_or_work_activity
income_context
relationship_role
status
state
preference
constraint
goal
plan
event_involvement
location
time_context
money_or_financial_context
sensitive_life_context
correction
denial
confirmation
possible_connection_between_entities
task_requirement
project_requirement
unknown_reference_signal
user_reaction
approval
rejection
irritation
frustration
impatience
dissatisfaction_with_complexity
request_to_simplify
```

## 13. Allowed signal scopes

```text
about_user
about_other_person
about_task
about_project
about_current_interaction
about_message_itself
about_unknown_entity
```

## 14. Stage 0 grounding rule

Every extracted entity and every extracted information signal must be grounded in a `source_span` from `current_user_message`.

If a signal depends on `recent_messages` only for reference resolution, the signal must still be present in `current_user_message`.

Do not create a signal from `recent_messages` alone.

## 15. Stage 0 relation rule

Stage 0 may mark that a message fragment appears to connect entities.

Stage 0 must not convert that fragment into a durable relation.

Correct:

```json
{
  "id": "s1",
  "source_span": "Алёна работает с нами",
  "signal_type": "possible_connection_between_entities",
  "about_entity_ids": ["e1", "e2"],
  "signal_scope": "about_other_person",
  "polarity": "asserted",
  "epistemic_status": "user_claim",
  "extraction_note": "The fragment appears to connect a person mention with a group or project context."
}
```

Incorrect:

```json
{
  "subject": "Алёна",
  "relation": "works_with",
  "object": "we"
}
```

Stage 0 preserves the surface signal.

Segment Analysis may later interpret it.

## 16. Stage 0 interaction reaction rule

Stage 0 may extract explicit user reactions when they are visible in the current message.

These include:

```text
approval
rejection
irritation
frustration
impatience
correction pressure
dissatisfaction with complexity
request to simplify
```

These are signals about the current interaction only.

They must not be treated as durable personality facts.

Correct:

```json
{
  "id": "s1",
  "source_span": "Какая у тебя большая бюрократия?",
  "signal_type": "dissatisfaction_with_complexity",
  "about_entity_ids": [],
  "signal_scope": "about_current_interaction",
  "polarity": "asserted",
  "epistemic_status": "user_reaction",
  "extraction_note": "The user explicitly rejects excessive process complexity."
}
```

Incorrect:

```json
{
  "claim": "User is an irritable person."
}
```

## 17. Stage 0 sensitive context rule

Do not sanitize, moralize, erase, or euphemize sensitive life context when the user explicitly provides it and it is relevant to understanding an entity, role, work, income, relationship, social context, risk, constraint, or situation.

If the user presents sex work or prostitution as work, income, biography, or social context, extract it neutrally as:

```text
work_activity
occupational_role
income_context
sensitive_life_context
```

Do not infer criminality, coercion, exploitation, trauma, danger, victimhood, promiscuity, moral failure, or risk unless explicitly stated or directly supported by the current message.

---

# Segment Analysis

## 18. Responsibility

Segment Analysis analyzes a dialogue segment, not a single isolated message.

It receives:

1. Raw turns.
2. Stage 0 outputs.
3. Previous dialogue analysis.

Segment Analysis interprets what the dialogue segment appears to mean.

Segment Analysis may identify:

```text
candidate facts
candidate corrections
candidate conflicts
candidate confirmations
important entities
open questions
do_not_save items
candidate memory items
```

Segment Analysis does not write durable memory.

Segment Analysis does not directly update durable memory.

Segment Analysis does not decide final save, update, skip, conflict, or ask status.

## 19. When Segment Analysis runs

Segment Analysis may run when one of these conditions is met:

```text
after 6–12 messages
after clear topic shift
after user correction
after task completion
after explicit user summary
after sensitive or identity-bearing signal
before memory write
```

Default V1 rule:

```text
Run Segment Analysis after every 6–12 messages or when the active topic changes.
```

## 20. Segment Analysis input

```json
{
  "dialogue_id": "",
  "track_id": "",
  "segment_id": "",
  "raw_turns": [],
  "stage0_signals": [],
  "previous_dialogue_analysis": null
}
```

## 21. Segment Analysis output

```json
{
  "segment_summary": "",
  "entities_in_segment": [],
  "interpreted_signals": [],
  "candidate_facts": [],
  "candidate_corrections": [],
  "candidate_conflicts": [],
  "candidate_memory_items": [],
  "open_questions": [],
  "do_not_save": [],
  "confidence": "low | medium | high"
}
```

## 22. Interpreted signal object

```json
{
  "id": "is1",
  "source_signal_ids": [],
  "source_turn_ids": [],
  "interpretation": "",
  "about_entity_ids": [],
  "interpretation_type": "",
  "confidence": "low | medium | high"
}
```

## 23. Candidate fact object

```json
{
  "id": "cf1",
  "claim": "",
  "about_entity_ids": [],
  "source_turn_ids": [],
  "source_signal_ids": [],
  "polarity": "asserted | negated | corrected | uncertain",
  "confidence": "low | medium | high"
}
```

Candidate facts are interpreted claims that may be true, false, corrected, uncertain, temporary, or context-bound.

A candidate fact is not automatically a memory candidate.

## 24. Candidate correction object

```json
{
  "id": "cc1",
  "corrected_claim": "",
  "replacement_claim": "",
  "about_entity_ids": [],
  "source_turn_ids": [],
  "source_signal_ids": [],
  "confidence": "low | medium | high"
}
```

## 25. Candidate conflict object

```json
{
  "id": "conf1",
  "conflict_description": "",
  "claims_in_conflict": [],
  "about_entity_ids": [],
  "source_turn_ids": [],
  "source_signal_ids": [],
  "confidence": "low | medium | high"
}
```

## 26. Candidate memory item object

```json
{
  "id": "cm1",
  "memory_type": "",
  "proposed_content": "",
  "about_entity_ids": [],
  "source_turn_ids": [],
  "source_signal_ids": [],
  "reason_for_candidate": "",
  "confidence": "low | medium | high"
}
```

Candidate memory items are proposals that some interpreted information may be useful for durable memory.

Not every candidate fact should become a candidate memory item.

Every candidate memory item must preserve source turn IDs and source signal IDs.

Segment Analysis candidate memory items are proposals only.

Memory Decision may reject any or all of them.

## 27. Do-not-save object

```json
{
  "id": "dns1",
  "content": "",
  "reason": "",
  "source_turn_ids": [],
  "source_signal_ids": [],
  "confidence": "low | medium | high"
}
```

Use `do_not_save` for information that may be useful for current interpretation but should not be proposed for durable memory.

Examples:

```text
current-turn formatting preferences
temporary frustration
one-off interaction reactions
procedural comments
low-confidence interpretations
sensitive information that is not necessary to store
```

## 28. Segment Analysis rules

Segment Analysis may interpret.

Segment Analysis may group multiple Stage 0 signals.

Segment Analysis may detect that a later message corrects an earlier message.

Segment Analysis may detect that the user confirmed or denied a previous interpretation.

Segment Analysis may propose memory candidates.

Segment Analysis must preserve source turn IDs and signal IDs.

Segment Analysis must not write memory.

Segment Analysis must not decide final memory action.

Segment Analysis must not treat weak signals as confirmed facts.

Segment Analysis must separate:

```text
what was said
what seems implied
what was corrected
what remains uncertain
what may be memory-worthy
what should not be saved
```

## 29. Segment Analysis user reaction handling

Segment Analysis may interpret repeated or explicit interaction reactions as current dialogue requirements.

Example:

```text
User says: "Какая у тебя большая бюрократия?"
```

Stage 0 signal:

```text
dissatisfaction_with_complexity
```

Segment Analysis interpretation:

```text
The user rejects the extra intermediate stage and wants a simpler pipeline.
```

Segment Analysis may produce a task requirement:

```json
{
  "id": "cf1",
  "claim": "The pipeline should not include an unnecessary intermediate Stage 1.",
  "about_entity_ids": [],
  "source_turn_ids": [],
  "source_signal_ids": [],
  "polarity": "asserted",
  "confidence": "high"
}
```

Segment Analysis must not produce:

```text
The user is generally irritable.
```

---

# Memory Decision

## 30. Responsibility

Memory Decision is the only stage allowed to decide what happens to durable memory.

Memory Decision receives candidate memory items and supporting evidence from Segment Analysis, including candidate facts, corrections, conflicts, source turn IDs, and source signal IDs.

Memory Decision loads durable memory.

Memory Decision compares candidates and supporting evidence against durable memory.

Memory Decision returns final memory decisions.

## 31. Memory Decision input

```json
{
  "candidate_memory_items": [],
  "candidate_facts": [],
  "candidate_corrections": [],
  "candidate_conflicts": [],
  "durable_memory_context": []
}
```

## 32. Memory Decision output

```json
{
  "decisions": []
}
```

## 33. Decision object

```json
{
  "candidate_id": "",
  "decision": "save | skip | update | conflict | ask",
  "target_memory_id": null,
  "reason": "",
  "final_content": null,
  "confidence": "low | medium | high"
}
```

## 34. Memory Decision rules

Memory Decision may compare against durable memory.

Memory Decision may decide that a candidate is already known.

Memory Decision may decide that a candidate updates an existing memory.

Memory Decision may decide that a candidate conflicts with existing memory.

Memory Decision may decide that the system must ask the user before saving.

Memory Decision may decide that nothing should be saved.

Memory Decision may reject any or all candidate memory items from Segment Analysis.

Memory Decision is the only durable-memory gate.

---

# Runtime flow

## 35. Normal message flow

```text
1. Save raw user turn.

2. Run Stage 0 on current_user_message.

3. Save current_signal with source spans.

4. Continue dialogue normally.

5. Do not write memory from Stage 0.
```

## 36. Segment analysis flow

```text
1. Detect segment boundary:
   - 6–12 messages passed;
   - topic changed;
   - user corrected something;
   - task completed;
   - sensitive or identity-bearing signal appeared;
   - memory write is being considered.

2. Load raw turns for the segment.

3. Load Stage 0 outputs for the segment.

4. Load previous dialogue analysis if available.

5. Run Segment Analysis.

6. Save Segment Analysis.

7. Send candidate memory items and supporting evidence to Memory Decision.
```

## 37. Memory decision flow

```text
1. Load candidate memory items from Segment Analysis.

2. Load supporting evidence from Segment Analysis.

3. Load relevant durable memory.

4. Compare candidates and evidence against durable memory.

5. Return decision:
   - save;
   - skip;
   - update;
   - conflict;
   - ask.

6. Only approved decisions affect durable memory.
```

---

# Data persistence

## 38. Required stored objects

The system should persist:

```text
raw_turn
stage0_current_signal
dialogue_segment_analysis
memory_candidate
memory_decision
memory_item
```

## 39. raw_turn

```json
{
  "turn_id": "",
  "dialogue_id": "",
  "track_id": "",
  "role": "user | assistant | tool",
  "content": "",
  "created_at": ""
}
```

## 40. stage0_current_signal

```json
{
  "signal_id": "",
  "turn_id": "",
  "dialogue_id": "",
  "track_id": "",
  "entities": [],
  "information_signals": [],
  "unresolved_references": [],
  "ambiguous_references": [],
  "created_at": ""
}
```

## 41. dialogue_segment_analysis

```json
{
  "segment_id": "",
  "dialogue_id": "",
  "track_id": "",
  "turn_ids": [],
  "stage0_signal_ids": [],
  "analysis": {},
  "created_at": ""
}
```

## 42. memory_candidate

```json
{
  "candidate_id": "",
  "segment_id": "",
  "dialogue_id": "",
  "track_id": "",
  "proposed_content": "",
  "source_turn_ids": [],
  "source_signal_ids": [],
  "confidence": "low | medium | high",
  "created_at": ""
}
```

## 43. memory_decision

```json
{
  "decision_id": "",
  "candidate_id": "",
  "decision": "save | skip | update | conflict | ask",
  "target_memory_id": null,
  "reason": "",
  "final_content": null,
  "confidence": "low | medium | high",
  "created_at": ""
}
```

---

# Non-negotiable rules

## 44. Stage boundaries

Stage 0 must never write memory.

Stage 0 must never decide novelty.

Stage 0 must never decide memory worthiness.

Stage 0 must never convert a mention into a durable fact.

Segment Analysis must never write durable memory.

Segment Analysis must always preserve source turn IDs and signal IDs.

Memory Decision must always compare candidates and supporting evidence against durable memory before writing.

## 45. Sensitive context

No stage may silently generalize sensitive context.

No stage may erase sensitive context if the user explicitly provided it and it is relevant to understanding the entity, situation, work, income, relationship, or constraint.

No stage may infer criminality, coercion, exploitation, trauma, danger, victimhood, promiscuity, moral failure, or moral judgment unless explicitly stated or directly supported by the dialogue segment.

## 46. User reaction signals

User reaction signals may be used to understand the current interaction.

They must not automatically become durable memory.

They must not become stable personality claims unless the user explicitly states a stable preference or repeated evidence is confirmed by Memory Decision.

---

# Minimal V1 implementation

## 47. V1 scope

V1 implements only this:

```text
Stage 0:
current message → entities + information_signals

Segment Analysis:
last 6–12 turns + Stage 0 outputs → candidate_facts + candidate_memory_items + do_not_save

Memory Decision:
candidate_memory_items + supporting evidence + durable memory → save / skip / update / conflict / ask
```

## 48. V1 exclusions

Do not add extra stages.

Do not add advanced ranking.

Do not add complex policy routing.

Do not add long-term inference.

Do not add automatic personality modeling.

Do not optimize before the pipeline works.

## 49. V1 acceptance criteria

The implementation is acceptable when:

```text
1. Raw user turns are always saved.

2. Stage 0 output is saved separately as current_signal.

3. Stage 0 output contains source spans.

4. Stage 0 does not create memory candidates.

5. Stage 0 does not write memory.

6. Stage 0 output cannot contain memory_candidates, selected_memory_ids, draft_answer, or final_answer.

7. Segment Analysis can analyze a segment using raw turns and Stage 0 outputs.

8. Segment Analysis can produce candidate_facts.

9. Segment Analysis can produce candidate_memory_items.

10. Segment Analysis can produce do_not_save items with reasons.

11. Segment Analysis preserves source turn IDs and signal IDs.

12. Memory Decision loads durable memory before deciding.

13. Memory Decision receives candidate memory items and supporting evidence.

14. Memory Decision returns save / skip / update / conflict / ask.

15. Memory Decision reason is required and non-empty.

16. Only Memory Decision can cause durable memory writes.

17. A bad candidate cannot roll back saved raw turns, assistant answers, or analysis audit records.

18. User irritation, rejection, correction pressure, and dissatisfaction with complexity can be captured as current-interaction signals.

19. These reaction signals are not stored as durable personality claims unless explicitly approved by Memory Decision.

20. Sensitive context can be represented neutrally as a signal without forced moralization or erasure.
```

---

# Codex implementation task

Implement Phase V1 contracts and tests for the dialogue signal and memory extraction pipeline.

Do not implement full memory logic yet.

## Goal

Create or update contracts for:

1. Stage 0: Current Signal Extraction
2. Segment Analysis, historically named Stage 2 if existing code already uses Stage2
3. Memory Decision

## Core architecture

```text
Stage 0 extracts only current-message entities and information_signals.

Stage 0 does not decide novelty, memory-worthiness, durable relations, or memory writes.

Segment Analysis interprets a dialogue segment and may propose candidate memory items.

Memory Decision is the only layer that may decide save, skip, update, conflict, or ask.

Durable memory writes must only happen after Memory Decision.
```

## Implementation scope

Implement only schemas, validators, and tests.

Do not implement full memory logic.

Do not implement memory write service changes unless needed to prove that Stage 0 and Segment Analysis cannot trigger writes.

## Required Stage 0 output shape

```json
{
  "entities": [],
  "information_signals": [],
  "unresolved_references": [],
  "ambiguous_references": []
}
```

## Required Stage 0 validation rules

Where practical, validate:

```text
Every entity must have an id.

Every information_signal must have an id.

Every information_signal must have source_span.

Every information_signal must preserve about_entity_ids as a list.

Stage 0 output must not contain memory_candidates.

Stage 0 output must not contain selected_memory_ids.

Stage 0 output must not contain final_answer or draft_answer.
```

## Required Segment Analysis output shape

```json
{
  "segment_summary": "",
  "entities_in_segment": [],
  "interpreted_signals": [],
  "candidate_facts": [],
  "candidate_corrections": [],
  "candidate_conflicts": [],
  "candidate_memory_items": [],
  "open_questions": [],
  "do_not_save": [],
  "confidence": "low | medium | high"
}
```

## Required Memory Decision output shape

```json
{
  "decisions": []
}
```

Each memory decision must contain:

```json
{
  "candidate_id": "",
  "decision": "save | skip | update | conflict | ask",
  "target_memory_id": null,
  "reason": "",
  "final_content": null,
  "confidence": "low | medium | high"
}
```

## Required tests

Add tests proving:

```text
1. Stage 0 accepts entities and information_signals.

2. Stage 0 rejects or ignores memory decision fields.

3. Stage 0 information_signals require source_span.

4. Segment Analysis can contain candidate_memory_items but does not write memory.

5. Memory Decision supports save, skip, update, conflict, and ask.

6. Memory Decision reason is required and non-empty.

7. Source turn IDs and source signal IDs are preserved in Segment Analysis candidate objects.

8. User reaction signals can be represented as current-interaction signals without becoming durable personality claims.

9. Sensitive context can be represented neutrally as a signal without forced moralization or erasure.

10. Memory writes are not triggered by Stage 0 or Segment Analysis contracts.

11. candidate_facts and candidate_memory_items are separate structures.

12. do_not_save items require reason and source references.

13. extraction_note is allowed on Stage 0 information signals but must not be required.

14. Stage 0 output does not include draft_answer or final_answer.
```

## Explicit exclusions

Do not add temporal metadata beyond existing model conventions.

Do not add validity scopes.

Do not add lifecycle operations.

Do not add ranking.

Do not add automatic personality modeling.

Do not add a runtime Stage 1.

Do not rename existing Stage2 code if that creates unnecessary churn; instead, document it as historical compatibility.
