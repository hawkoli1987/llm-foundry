
s3_files=(
    "s3://seafm-data/sea-lion-pile/tl/raw/train.jsonl"
    "s3://seafm-data/wiki/km/kmwiki_dedup.jsonl"
    "s3://seafm-data/wiki/km/kmwikibooks_dedup.jsonl"
    "s3://seafm-data/news/km/globalvoice/cleaned/gv_km_dedup.jsonl"
    "s3://seafm-data/news/km/voice_of_america/clean/voa_km_dedup.jsonl"
    "s3://seafm-data/wiki/lo/lowiki_dedup.jsonl"
    "s3://seafm-data/news/lo/voice_of_america/cleaned/voa_lo_dedup.jsonl"
    "s3://seafm-data/wiki/ms/mswiki_dedup.jsonl"
    "s3://seafm-data/wiki/ms/mswikibooks_dedup.jsonl"
    "s3://seafm-data/wiki/ta/tawiki_dedup.jsonl"
    "s3://seafm-data/wiki/ta/tawikibooks_dedup.jsonl"
    "s3://seafm-data/wiki/ta/tawikinews_dedup.jsonl"
    "s3://seafm-data/wiki/ta/tawikisource_dedup.jsonl"
    "s3://seafm-data/news/ta/seithi/cleaned/seithi_dedup.jsonl"
    "s3://seafm-data/wiki/th/thwiki_dedup.jsonl"
    "s3://seafm-data/wiki/th/thwikibooks_dedup.jsonl"
    "s3://seafm-data/wiki/th/thwikisource_dedup.jsonl"
    "s3://seafm-data/news/th/voice_of_america/clean/voa_th_dedup.jsonl"
    "s3://seafm-data/wiki/tl/tlwiki_dedup.jsonl"
    "s3://seafm-data/wiki/tl/tlwikibooks_dedup.jsonl"
    "s3://seafm-data/news/tl/globalvoice/cleaned/gv_tl_dedup.jsonl"
    "s3://seafm-data/wiki/vi/viwiki_dedup.jsonl"
    "s3://seafm-data/wiki/vi/viwikibooks_dedup.jsonl"
    "s3://seafm-data/wiki/vi/viwikisource_dedup.jsonl"
    "s3://seafm-data/wiki/vi/viwikivoyage_dedup.jsonl"
    "s3://seafm-data/news/vi/voice_of_america/clean/voa_vi_dedup.jsonl"
    "s3://seafm-data/wiki/zh/zhwiki_dedup.jsonl"
    "s3://seafm-data/wiki/zh/zhwikibooks_dedup.jsonl"
    "s3://seafm-data/wiki/zh/zhwikinews_dedup.jsonl"
    "s3://seafm-data/wiki/zh/zhwikisource_dedup.jsonl"
    "s3://seafm-data/wiki/zh/zhwikivoyage_dedup.jsonl"
    "s3://seafm-data/news/zh/8world/cleaned/8world_dedup.jsonl"
    "s3://seafm-data/news/zh/voice_of_america/clean/voa_zh_dedup.jsonl"
)
local_dir=/gojek/data/raw/
mkdir -p "$local_dir"
for s3_file in "${s3_files[@]}"; do
    aws s3 cp \
    "$s3_file" \
    "$local_dir" \
    --endpoint-url https://s3-accelerate.amazonaws.com
done