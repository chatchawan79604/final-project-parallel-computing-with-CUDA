#include <stdio.h>
#include <stdlib.h>

#define MAX_LEN 7

int max_id(int corpus[][MAX_LEN], size_t corpus_size, size_t max_len)
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
  return max;
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
      {2, 5, 3, 3, 0, -1, -1},
      {2, 5, 4, 4, 1, 1, 1},
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

  int vocab_size = max_id(corpus, corpus_size, 7);
  printf("%d\n", vocab_size);

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

  printf("tf\n");
  for (int i = 0; i < corpus_size; i++)
  {
    for (size_t j = 0; j < vocab_size; j++)
    {
      printf("%d, ", counts[i][j]);
    }
    printf("\n");
  }
  printf("\n");

  putc('\n', stdout);
  return 0;
}
