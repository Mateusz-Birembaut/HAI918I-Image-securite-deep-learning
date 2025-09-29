#!/bin/bash
set -euo pipefail

# detect available png terminal (prefer pngcairo)
if gnuplot -e "set terminal" 2>/dev/null | grep -qi pngcairo; then
    TERM="pngcairo"
else
    TERM="png"
fi

gnuplot <<EOF
set terminal ${TERM} size 1000,480
set output 'hist_compare.png'
set title 'Histogramme: original vs chiffré (ECB)'
set style fill solid 0.6
set boxwidth 0.4
set xlabel 'Pixel value'
set ylabel 'Count'
set xtics 0,25,255
plot 'M_Hist.dat' every ::1 using 1:2 with boxes fc rgb 'blue' title 'original', \
     'M_ECB_Hist.dat' every ::1 using (\$1+0.4):2 with boxes fc rgb 'red' title 'ECB'
EOF