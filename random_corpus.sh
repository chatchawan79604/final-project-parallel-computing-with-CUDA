#!/bin/bash

n=128    # maximum random number (exclusive)
x=32     # numbers per line
y=24     # number of lines
filename="corpus.txt"

rm -f "$filename"    # remove the file if it already exists

for ((i=1; i<=y; i++))
do
    seq=$((RANDOM % x))
    for ((j=1; j<=seq; j++))
    do
        echo -n "$((RANDOM % n)) " >> "$filename"
    done
    echo >> "$filename"
done
