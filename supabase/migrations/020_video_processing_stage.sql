-- Real pipeline stage shown in the UI instead of time-based fake progress.
alter table videos
add column if not exists processing_stage text;
