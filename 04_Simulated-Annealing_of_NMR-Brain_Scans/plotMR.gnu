###gnuplot commands:
unset key
set size 0.26,0.5
#set pm3d map
#set dgrid3d 254,333
set palette gray
unset xtics
unset ytics
unset colorbox
#set xrange [1:254]
#set yrange [1:333] 
plot "SimMRimage.dat"  u 1:(-$2):3 with image t ""



 se term postscript eps enhanced color
 se output "SimMRimage.ps"
 replot
