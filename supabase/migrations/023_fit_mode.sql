-- Add Fit / Fill mode to exports and cached clip edits.

alter table exports
  add column if not exists fit_mode text not null default 'fill'
  check (fit_mode in ('fill','fit'));

alter table clip_edits
  add column if not exists fit_mode text not null default 'fill'
  check (fit_mode in ('fill','fit'));

drop index if exists clip_edits_key_idx;

create unique index clip_edits_key_idx
  on clip_edits (clip_id, aspect_ratio, fit_mode, cuts_hash);
