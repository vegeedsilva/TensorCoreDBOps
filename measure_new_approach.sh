#!/bin/bash
for (( i =32; i <= 268435456; i*=2 ))
do
	printf $i
	printf ', '
	./select-baseline $i
	printf ' \t '
	./select-tensor-chunking-newapproach_chunk $i 256 256
	printf ', '
	./select-tensor-chunking-newapproach_chunk $i 512 256
	printf ', '
		./select-tensor-chunking-newapproach_chunk $i 1024 256
	printf ', '
		./select-tensor-chunking-newapproach_chunk $i 2048 256
	printf ' \n '
done
