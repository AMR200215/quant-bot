"""
research/spool — local durability layer for failed / stripped Supabase writes.

dropped_fields.jsonl  — one entry per field stripped by PGRST204 retry loop
failed_inserts.jsonl  — one entry per base row that could not be inserted at all

Both files are append-only.  replay_spool.py replays them after schema is fixed.
"""
