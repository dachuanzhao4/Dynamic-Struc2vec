
set -e

YEARS=(1980 1985 1990 1995 2000 2005 2010)

mkdir -p results

for Y in "${YEARS[@]}"; do
  echo "=== Running test for snapshot $Y ==="
  python src/test.py \
    --dist-prefix "$Y" \
    --embedding-file "emb/${Y}.emb" \
    --id-type int \
    > "results/${Y}_test.log" 2>&1
  echo "    → results written to results/${Y}_test.log"
done
