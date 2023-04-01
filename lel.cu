#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define MAX_LINE_LENGTH 1024

__global__ void find_vocab_size_kernel(int *corpus, int *out_arr, size_t corpus_size, size_t max_len)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ int sdata[256];
  if (idx < corpus_size * max_len)
  {
    // load data into shared memory
    sdata[threadIdx.x] = corpus[idx];
    __syncthreads();

    // sequential addressing reduction
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
      if (threadIdx.x < stride)
      {
        // compare and replace
        int lv = sdata[threadIdx.x];
        int rv = sdata[threadIdx.x + stride];
        if (lv < rv)
        {
          sdata[threadIdx.x] = rv;
        }
        else
        {
          sdata[threadIdx.x] = lv;
        }
      }
      __syncthreads();
    }
  }

  // write result for this block to global memory
  if (threadIdx.x == 0)
  {
    out_arr[blockIdx.x] = sdata[0];
  }
}

int find_vocab_size(int *corpus, size_t corpus_size, size_t max_len)
{
  if (corpus == NULL)
  {
    fprintf(stderr, "corpus is null\n");
    return -1;
  }

  int blocksPerGrid = (corpus_size * max_len + 511) / 512;
  int *d_corpus, *d_max;
  cudaMalloc((void **)&d_corpus, sizeof(int) * corpus_size * max_len);
  cudaMemcpy(d_corpus, corpus, sizeof(int) * corpus_size * max_len, cudaMemcpyHostToDevice);

  int *h_max;
  h_max = (int *)malloc(sizeof(int) * blocksPerGrid);
  cudaMalloc((void **)&d_max, sizeof(int) * blocksPerGrid);
  cudaMemcpy(d_max, h_max, sizeof(int) * blocksPerGrid, cudaMemcpyHostToDevice);

  find_vocab_size_kernel<<<blocksPerGrid, 256>>>(d_corpus, d_max, corpus_size, max_len);
  cudaMemcpy(h_max, d_max, sizeof(int) * blocksPerGrid, cudaMemcpyDeviceToHost);
  cudaFree(d_corpus);
  cudaFree(d_max);

  int max = h_max[0];
  for (size_t i = 0; i < blocksPerGrid; i++)
  {
    if (max < h_max[i])
    {
      max = h_max[i];
    }
  }

  return max + 1;
}

void frequency_count(int *corpus, int corpus_size, int max_len, int vocab_size, int *count_arr)
{
  for (size_t i = 0; i < corpus_size; i++)
  {
    for (size_t j = 0; j < max_len; j++)
    {
      if (corpus[i * max_len + j] != -1)
      {
        count_arr[i * vocab_size + corpus[i * max_len + j]]++;
      }
    }
  }
}

// TODO: optimize this part
__global__ void term_frequency_kernel(int *term_counts, int corpus_size, int vocab_size, double *out_arr)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < corpus_size)
  {
    int wc_sum = 0;
    for (size_t wj = 0; wj < vocab_size; wj++)
    {
      wc_sum += term_counts[i * vocab_size + wj];
    }

    for (size_t j = 0; j < vocab_size; j++)
    {
      out_arr[i * vocab_size + j] = (double)term_counts[i * vocab_size + j] / wc_sum;
    }
  }
}

void term_frequency(int *term_counts, int corpus_size, int vocab_size, double *tf_arr)
{
  int *d_term_counts;
  double *d_tf_arr;

  cudaMalloc(&d_term_counts, corpus_size * vocab_size * sizeof(int));
  cudaMemcpy(d_term_counts, term_counts, corpus_size * vocab_size * sizeof(int), cudaMemcpyHostToDevice);

  cudaMalloc(&d_tf_arr, corpus_size * vocab_size * sizeof(double));

  int block_size = 256;
  int num_blocks = (corpus_size + block_size - 1) / block_size;

  term_frequency_kernel<<<num_blocks, block_size>>>(d_term_counts, corpus_size, vocab_size, d_tf_arr);

  cudaMemcpy(tf_arr, d_tf_arr, corpus_size * vocab_size * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_term_counts);
  cudaFree(d_tf_arr);
}


void invert_document_frequency(int *term_counts, int corpus_size, int vocab_size, int smooth_idf, double *out_arr)
{
  for (size_t wi = 0; wi < vocab_size; wi++)
  {
    int w_dc = 0 + smooth_idf;
    for (size_t di = 0; di < corpus_size; di++)
    {
      if (term_counts[di * vocab_size + wi])
        w_dc++;
    }

    out_arr[wi] = log((double)corpus_size / w_dc) + 1;
  }
}

void tfidf(double *tf_arr, double *idf_arr, double *out_arr, int corpus_size, int vocab_size)
{
  for (int di = 0; di < corpus_size; di++)
  {
    for (size_t wi = 0; wi < vocab_size; wi++)
    {
      out_arr[di * vocab_size + wi] = tf_arr[di * vocab_size + wi] * idf_arr[wi];
    }
  }
}

void read_corpus(char *filename, int **array, int *num_rows, int *num_cols)
{
  FILE *fp;
  size_t line_size = MAX_LINE_LENGTH;
  char line[line_size];
  int num_columns = 0, num_lines = 0;
  *num_cols = 0;
  *num_rows = 0;

  fp = fopen(filename, "r");
  if (fp == NULL)
  {
    printf("Error: could not open file %s\n", filename);
    return;
  }

  while (fgets(line, line_size, fp) != NULL)
  {
    num_columns = 0;
    num_lines++;
    char *token = strtok(line, " ");
    while (token != NULL)
    {
      if (strcmp(token, "\n") != 0)
        num_columns++;
      token = strtok(NULL, " ");
    }
    if (num_columns > *num_cols)
    {
      *num_cols = num_columns;
    }
    if (num_columns > 0)
    {
      (*num_rows)++;
    }
  }
  *array = (int *)malloc(*num_rows * *num_cols * sizeof(int));
  fseek(fp, 0, SEEK_SET);
  int i = 0, j;
  while (fgets(line, line_size, fp) != NULL)
  {
    j = 0;
    char *token = strtok(line, " ");
    while (token != NULL)
    {
      (*array)[i * *num_cols + j] = atoi(token);
      j++;
      token = strtok(NULL, " ");
    }
    while (j < *num_cols)
    {
      (*array)[i * *num_cols + j] = -1; // Fill the remaining columns with padding
      j++;
    }
    i++;
  }

  fclose(fp);
}

int main()
{
  int smooth_idf = 0; // should add 1 or not
  int vocab_size = 0, seq_max_len = 0, corpus_size = 0;
  int *corpus;

  // read input corpus from text file
  char corpus_file_name[] = "corpus.txt";
  read_corpus(corpus_file_name, &corpus, &corpus_size, &seq_max_len);
  printf("seq_max_len: %d, corpus size: %d\n", seq_max_len, corpus_size);

  // find maximum word id
  vocab_size = find_vocab_size(corpus, corpus_size, seq_max_len);
  printf("vocab size: %d\n", vocab_size);

  // initailize word counts array
  int *counts = (int *)malloc(sizeof(int) * corpus_size * vocab_size);
  for (int i = 0; i < corpus_size; i++)
  {
    for (size_t j = 0; j < vocab_size; j++)
    {
      // set to identity
      counts[i * vocab_size + j] = 0;
    }
  }

  // initailize tf array
  double *tf_arr = (double *)malloc(sizeof(double *) * corpus_size * vocab_size);

  // initailize idf array
  double *idf_arr = (double *)malloc(sizeof(double) * vocab_size);

  frequency_count(corpus, corpus_size, seq_max_len, vocab_size, counts);
  term_frequency(counts, corpus_size, vocab_size, tf_arr);
  invert_document_frequency(counts, corpus_size, vocab_size, smooth_idf, idf_arr);
  tfidf(tf_arr, idf_arr, tf_arr, corpus_size, vocab_size);

  for (size_t i = 0; i < corpus_size; i++)
  {
    for (size_t j = 0; j < vocab_size; j++)
    {
      printf("%.3f ", tf_arr[i * vocab_size + j]);
    }
    printf("\n");
  }

  free(idf_arr);
  free(tf_arr);
  free(counts);
  free(corpus);
  return 0;
}
