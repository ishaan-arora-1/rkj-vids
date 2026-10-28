#!/bin/bash
id=$1; clip=$2; ss=$3; dur=$4; sp=$5
src=/home/user/rkj-vids/IMG_$clip.MP4
ffmpeg -nostdin -v error -y -ss $ss -i $src -t $dur -an -c:v libx264 -crf 8 -preset fast $id.raw.mp4
ffmpeg -nostdin -v error -y -i $id.raw.mp4 -vf vidstabdetect=shakiness=6:accuracy=15:result=$id.trf -f null -
ffmpeg -nostdin -v error -y -i $id.raw.mp4 -vf "vidstabtransform=input=$id.trf:smoothing=25:zoom=4:optzoom=0:interpol=bicubic,setpts=PTS/$sp,minterpolate=fps=30:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1" -c:v libx264 -crf 8 -preset fast $id.mp4
