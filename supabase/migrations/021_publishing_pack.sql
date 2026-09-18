-- Publishing Pack for each finished Moment.
alter table clips
  add column if not exists social_title_options jsonb not null default '[]',
  add column if not exists social_description_options jsonb not null default '[]',
  add column if not exists overlay_text_options jsonb not null default '[]',
  add column if not exists overlay_text text;
