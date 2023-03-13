#!/bin/bash

for cat in pan 
  do
	for trial in 3 4
	    do
		for i in {0..19}
		do
		    python sample_grasps_semantic.py --trial $trial --cat $cat
		done 
	    done

  done
