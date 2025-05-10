set -e

CONF="src/conf/timeline.yml"

YEARS=(1985 1990 1995 2000 2005 2010)

for Y in "${YEARS[@]}"; do
  if [[ "$Y" == "1985" ]]; then
    echo "=== Incremental Warm-Start for $Y (base on 1980) ==="
    python src/sd_loop.py --conf "$CONF" --step "$Y"
  else
    echo "=== Incremental Warm-Start for $Y (base on ${prev}_warm) ==="
    python src/sd_loop.py --conf "$CONF" --step "$Y" --use-warm
  fi
  prev=$Y
done