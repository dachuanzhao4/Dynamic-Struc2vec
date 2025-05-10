set -e

YEARS=(1985 1990 1995 2000 2005 2010)

mkdir -p results

for Y in "${YEARS[@]}"; do
  echo "=== Running test for warm snapshot ${Y}_warm ==="
  python src/test.py \
    --dist-prefix "${Y}_warm" \
    --embedding-file "emb/${Y}_warm.emb" \
    --id-type int \
    > "results/${Y}_warm_test.log" 2>&1
  echo "    → results written to results/${Y}_warm_test.log"
done
