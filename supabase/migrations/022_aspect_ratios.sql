-- Add the new clip/export aspect ratios.

alter table exports
  drop constraint if exists exports_aspect_ratio_check;

alter table exports
  add constraint exports_aspect_ratio_check
  check (aspect_ratio in ('9:16','16:9','1:1','4:5','3:4','4:3'));

alter table clip_edits
  drop constraint if exists clip_edits_aspect_ratio_check;

alter table clip_edits
  add constraint clip_edits_aspect_ratio_check
  check (aspect_ratio in ('9:16','16:9','1:1','4:5','3:4','4:3'));
