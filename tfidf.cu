#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define MAX_LINE_LENGTH 2048

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

__global__ void frequency_count_kernel(int *corpus, int *count_arr, int seq_max_len, int vocab_size, int corpus_size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < corpus_size * seq_max_len)
  {
    // load data into shared memory
    
    for (int count = seq_max_len; count > 0; count = count / 2)
  {
    if (idx < seq_max_len)
    {
      printf("count: %d  thread: %d\n", count, idx);
      corpus[idx] = corpus[idx] + corpus[idx + count];
    }
    seq_max_len /= 2;
  }
  if (idx == 0)
    count_arr[0] = corpus[0];
  }
}

void frequency_count(int *corpus, int corpus_size, int seq_max_len, int vocab_size, int *count_arr)
{
  int blocksPerGrid = (corpus_size * seq_max_len + 511) / 512;
  int *d_corpus, *d_count_arr;
  cudaMalloc((void **)&d_corpus, sizeof(int) * corpus_size * seq_max_len);
  cudaMemcpy(d_corpus, corpus, sizeof(int) * corpus_size * seq_max_len, cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_count_arr, sizeof(int) * corpus_size * vocab_size);
  cudaMemcpy(d_count_arr, count_arr, sizeof(int) * corpus_size * vocab_size, cudaMemcpyHostToDevice);
  d_count_arr = (int *)malloc(sizeof(int) * corpus_size * vocab_size);
  
  int *h_max;
  h_max = (int *)malloc(sizeof(int) * blocksPerGrid);
  cudaMalloc((void **)&d_corpus, sizeof(int) * blocksPerGrid);
  cudaMemcpy(d_corpus, h_max, sizeof(int) * blocksPerGrid, cudaMemcpyHostToDevice);
  int *h_count_arr;
  h_count_arr = (int *)malloc(sizeof(int) * corpus_size * vocab_size);
  cudaMalloc((void **)&d_count_arr, sizeof(int) * corpus_size * vocab_size);
  cudaMemcpy(d_count_arr, h_count_arr, sizeof(int) * corpus_size * vocab_size, cudaMemcpyHostToDevice);

  // printf("corpus_size=%d, seq_max_len=%d, vocab_size=%d\n", corpus_size, seq_max_len, vocab_size);
  frequency_count_kernel<<<blocksPerGrid, 256>>>(d_corpus, d_count_arr, seq_max_len,vocab_size, corpus_size);
  cudaMemcpy(count_arr, d_count_arr, sizeof(int) * corpus_size * vocab_size, cudaMemcpyDeviceToHost);
  cudaFree(d_corpus);
  cudaFree(d_count_arr);

}

__global__ void term_frequency_kernel(int *term_counts, int corpus_size, int vocab_size, double *out_arr)
{
  int doc = blockIdx.x;
  int wrd = threadIdx.x;
  __shared__ int scounts[256];

  if (wrd < vocab_size)
  {
    scounts[wrd] = term_counts[doc * vocab_size + wrd];
    __syncthreads();

    for (int stride = 1; stride < vocab_size; stride *= 2)
    {
      int index = 2 * stride * wrd;
      if (index < vocab_size)
      {
        scounts[index] += scounts[index + stride];
      }
      __syncthreads();
    }

    // if (wrd == 0)
    //   printf("doc(%d) => %d\n", doc, scounts[0]);

    out_arr[doc * vocab_size + wrd] = (double)term_counts[doc * vocab_size + wrd] / scounts[0];

    // printf("tf(doc=%d,wrd=%d) = %f\n", doc, wrd, out_arr[doc * vocab_size + wrd]);
  }
}

__global__ void invert_document_frequency_kernel(int *term_counts, int corpus_size, int vocab_size, int smooth_idf, double *out_arr)
{
  int wrd = blockIdx.x;
  int doc = threadIdx.x;
  __shared__ int scounts[256];

  if (wrd < vocab_size)
  {
    scounts[doc] = term_counts[doc * vocab_size + wrd] ? 1 : 0;
    __syncthreads();

    for (int stride = (corpus_size + 1) / 2; stride > 0; stride >>= 1)
    {
      if (doc < stride)
      {
        // printf("stride%d: %d+%d\n", stride, scounts[wrd], scounts[wrd + stride]);
        scounts[doc] += scounts[doc + stride];
      }
      __syncthreads();
    }
  }

  int w_dc = scounts[0] + smooth_idf;
  out_arr[wrd] = log10((double)corpus_size / w_dc);

  // if (doc == 0)
  // {
  //   printf("corpus_size = %d\n", corpus_size);
  //   printf("idf(wrd=%d) = %f\n", wrd, out_arr[wrd]);
  //   printf("w_doc_count(wrd=%d) = %d\n", wrd, w_dc);
  // }
}

__global__ void tfidf_kernel(const double *tf_arr, const double *idf_arr, double *out_arr, int corpus_size, int vocab_size)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < corpus_size * vocab_size)
  {
    int wi = idx % vocab_size;
    // int di = idx / vocab_size;
    out_arr[idx] = tf_arr[idx] * idf_arr[wi];
  }
}

void tfidf(int *term_counts, double *out_arr, int corpus_size, int vocab_size)
{
  int smooth_idf = 0;
  size_t num_elements = corpus_size * vocab_size;
  double *d_tf_arr, *d_idf_arr, *d_out_arr;
  int *d_term_counts;

  cudaMalloc((void **)&d_term_counts, corpus_size * vocab_size * sizeof(int));
  cudaMemcpy(d_term_counts, term_counts, corpus_size * vocab_size * sizeof(int), cudaMemcpyHostToDevice);

  /* term frequency */
  cudaMalloc((void **)&d_tf_arr, num_elements * sizeof(double));
  int tf_block_size = vocab_size;
  int tf_num_blocks = corpus_size;
  term_frequency_kernel<<<tf_num_blocks, tf_block_size>>>(d_term_counts, corpus_size, vocab_size, d_tf_arr);

  /* inverted document frequency */
  cudaMalloc((void **)&d_idf_arr, vocab_size * sizeof(double));
  int idf_block_size = corpus_size;
  int idf_num_blocks = vocab_size;
  invert_document_frequency_kernel<<<idf_num_blocks, idf_block_size>>>(d_term_counts, corpus_size, vocab_size, smooth_idf, d_idf_arr);
  cudaFree(d_term_counts);

  /* tfidf */
  cudaMalloc((void **)&d_out_arr, num_elements * sizeof(double));
  int tfidf_block_size = 256;
  int tfidf_num_blocks = (num_elements + tfidf_block_size - 1) / tfidf_block_size;
  tfidf_kernel<<<tfidf_num_blocks, tfidf_block_size>>>(d_tf_arr, d_idf_arr, d_out_arr, corpus_size, vocab_size);

  cudaMemcpy(out_arr, d_out_arr, num_elements * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_tf_arr);
  cudaFree(d_idf_arr);
  cudaFree(d_out_arr);
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
    int line_len = strlen(line);
    for (int c = line_len - 1; c > 0; c--)
    {
      if (line[c] == ' ' || line[c] == '\n')
      {
        line[c] = '\0';
      }
      else
      {
        break;
      }
    }
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

void write_array_to_file(double *arr, int rows, int cols, const char *filename)
{
  FILE *f = fopen(filename, "w");
  if (f == NULL)
  {
    printf("Error opening file %s\n", filename);
    return;
  }

  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      fprintf(f, "%lf ", arr[i * cols + j]);
    }
    fprintf(f, "\n");
  }

  fclose(f);
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
  
  frequency_count(corpus, corpus_size, seq_max_len, vocab_size, counts);
  tfidf(counts, tf_arr, corpus_size, vocab_size);

  char tfidf_filename[] = "_tfidf.arr";
  write_array_to_file(tf_arr, corpus_size, vocab_size, tfidf_filename);

  free(tf_arr);
  free(counts);
  free(corpus);
  return 0;
}
