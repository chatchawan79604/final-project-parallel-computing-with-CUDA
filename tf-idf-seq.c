#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_LEN 7

int find_vocab_size(int corpus[][MAX_LEN], size_t corpus_size, size_t max_len)
{
  if (corpus == NULL)
  {
    fprintf(stderr, "corpus is null\n");
    return -1;
  }

  int max = corpus[0][0];
  for (size_t i = 0; i < corpus_size; i++)
  {
    for (size_t j = 0; j < max_len; j++)
    {
      if (corpus[i][j] > max)
        max = corpus[i][j];
      if (corpus[i][j + 1] == -1)
        break;
    }
  }
  return max + 1;
}

void frequency_count(int corpus[][MAX_LEN], size_t corpus_size, size_t max_len, int **count_arr)
{
  for (size_t i = 0; i < corpus_size; i++)
  {
    for (size_t j = 0; j < max_len; j++)
    {
      int wid = corpus[i][j];
      if (wid != -1)
      {
        count_arr[i][wid]++;
      }
    }
  }
}

int main()
{
  int corpus_size = 2;
  int corpus[][MAX_LEN] = {
      {3, 0, 1, 1, 5, -1, -1},
      {3, 0, 2, 2, 4, 4, 4},
  };

  printf("input corpus\n");
  for (size_t i = 0; i < corpus_size; i++)
  {
    for (size_t j = 0; j < MAX_LEN; j++)
    {
      printf("%d ", corpus[i][j]);
    }
    printf("\n");
  }

  int vocab_size = find_vocab_size(corpus, corpus_size, 7);
  printf("vocab size: %d\n", vocab_size);

  int **counts = (int **)malloc(sizeof(int *) * corpus_size);
  for (int i = 0; i < corpus_size; i++)
  {
    counts[i] = (int *)malloc(sizeof(int) * vocab_size);
    for (size_t j = 0; j < vocab_size; j++)
    {
      counts[i][j] = 0;
    }
  }

  frequency_count(corpus, corpus_size, MAX_LEN, counts);

  printf("word counts in each doc:\n");
  for (int i = 0; i < corpus_size; i++)
  {
    for (size_t j = 0; j < vocab_size; j++)
    {
      printf("%d, ", counts[i][j]);
    }
    printf("\n");
  }
  printf("\n");

  double **tf_arr = (double **)malloc(sizeof(double *) * corpus_size);
  for (int i = 0; i < corpus_size; i++)
  {
    int wc_sum = 0;
    for (size_t wj = 0; wj < vocab_size; wj++)
    {
      wc_sum += counts[i][wj];
    }

    tf_arr[i] = (double *)malloc(sizeof(double) * vocab_size);
    for (size_t j = 0; j < vocab_size; j++)
    {
      tf_arr[i][j] = (double)counts[i][j] / wc_sum;
    }
  }

  printf("tf:\n");
  for (int i = 0; i < corpus_size; i++)
  {
    for (size_t j = 0; j < vocab_size; j++)
    {
      printf("%.2f, ", tf_arr[i][j]);
    }
    printf("\n");
  }
  printf("\n");

  double *idf_arr = (double *)malloc(sizeof(double) * vocab_size);
  for (size_t wi = 0; wi < vocab_size; wi++)
  {
    int w_dc = 0;
    for (size_t di = 0; di < corpus_size; di++)
    {
      if (counts[di][wi])
        w_dc++;
    }

    idf_arr[wi] = log10((double)corpus_size / w_dc);
  }

  printf("idf:\n");
  for (size_t wi = 0; wi < vocab_size; wi++)
  {
    printf("%.3f, ", idf_arr[wi]);
  }
  printf("\n");

  putc('\n', stdout);
  return 0;
}
