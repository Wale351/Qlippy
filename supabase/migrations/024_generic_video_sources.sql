-- Generic video source support.
-- Keep legacy YouTube columns for backward compatibility.

alter table videos
  add column if not exists source_url text,
  add column if not exists source_type text,
  add column if not exists source_id text;

-- Backfill existing YouTube records.
update videos
set
  source_url  = coalesce(source_url, youtube_url),
  source_type = coalesce(source_type, 'youtube'),
  source_id   = coalesce(source_id, youtube_id)
where source_url is null
   or source_type is null
   or source_id is null;

-- New non-YouTube records must not be forced to have YouTube fields.
alter table videos
  alter column youtube_url drop not null,
  alter column youtube_id drop not null;

-- Prevent the same source item being added twice by the same user.
create unique index if not exists videos_user_source_idx
  on videos (user_id, source_type, source_id)
  where source_id is not null;
