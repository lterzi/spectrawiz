spectrawiz-process \
  --date 20260610 \
  --raw_path /project/meteo/data/mrr/incoming/202606/20260610/ \
  --output_dir processed_data/mrr/ \
  --radar_type mrr \
  --preprocessing_path /project/meteo/data/mrr/incoming/202601/20260101/ \
  --preprocessing_pattern "*.nc" \
  --hourly \
  --debugging \
  --overwrite