
set -e

CONF="src/conf/timeline.yml"

YEARS=(1985 1990 1995 2000 2005 2010)

for Y in "${YEARS[@]}"; do

    echo "=== Incremental Warm-Start for $Y (base on ${prev}) ==="
    python src/sd_loop.py --conf "$CONF" --step "$Y"
  prev=$Y
done