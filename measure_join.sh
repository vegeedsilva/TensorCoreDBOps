#!/bin/bash
for (( i =16; i <= 16384 ; i*=2 ))
do
	printf $i
	printf ', '
	./join-without-tensor $i
	printf ', '
	./join-without-tensor-chunking $i 512
	printf ', '
		./join-tensor $i
	printf ', '
		./join-tensor-chunking $i 512
    printf ', '
       ./join-baseline $i
	printf ' \n '
done
