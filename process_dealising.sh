spectrawiz-process --date 20250604 --raw_path /project/meteo/data/kamacs/mom/2025/06/04/ \
  --output_dir /scratch/l/L.Terzi/radar_processed/ --hourly --include_moments --include_ldr \
  --file_pattern '20250604_18*.znc' \
  --dealias_method ridge \
  --dealias_ref_pattern '/project/meteo/data/xmacs/mom/2025/06/04/20250604_18*.znc' \
  --start_time 18:00 --end_time 18:10 --dealias_n_jobs 4 \
  --overwrite --debugging
