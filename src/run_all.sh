
set -e

YEARS=(1980 1985 1990 1995 2000 2005 2010)

DIM=128
WL=80
NW=20
WS=10    
LAYER=5
OPT1="True"
OPT2="True"
OPT3="True"

for Y in "${YEARS[@]}"; do
  echo "=== Running Struc2vec static for snapshot $Y ==="
  python src/main.py \
    --input       "graph/edgelist${Y}.edgelist" \
    --output      "emb/${Y}.emb" \
    --prefix      "${Y}" \
    --dimensions  $DIM \
    --walk-length $WL \
    --num-walks   $NW \
    --window-size $WS \
    --until-layer $LAYER \
    --OPT1        $OPT1 \
    --OPT2        $OPT2 \
    --OPT3        $OPT3
done
