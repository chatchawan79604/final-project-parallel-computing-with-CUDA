run: tfidf.out
	./tfidf.out

tfidf.out: tfidf.cu
	nvcc tfidf.cu -o tfidf.out

clean:
	rm *.out