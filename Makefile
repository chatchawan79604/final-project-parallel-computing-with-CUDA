run: tfidf.out corpus.txt
	./tfidf.out

tfidf.out: tfidf.cu
	nvcc tfidf.cu -o tfidf.out

corpus.txt: random_corpus.sh
	./random_corpus.sh

prof: tfidf.out
	nvprof --print-gpu-trace ./tfidf.out

clean:
	rm *.out