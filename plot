set view equal xyz
set pointsize 0.5
set ticslevel .1
splot "data" every :::0::0 with points pointtype 2, "data" every :::1::1 with points pointtype 7, "data" every :::2::2 with points pointtype 4, "data" every :::3::3 with points pointtype 5
#splot "data" every 2 with points pointtype 5, "data" every 2::1 with points pointtype 7
