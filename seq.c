#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define MAX_LINE_LENGTH 1024

int find_vocab_size(int *corpus, size_t corpus_size, size_t max_len)
{
  if (corpus == NULL)
  {
    fprintf(stderr, "corpus is null\n");
    return -1;
  }

  int max = corpus[0];
  for (size_t i = 0; i < corpus_size; i++)
  {
    for (size_t j = 0; j < max_len; j++)
    {
      if (corpus[i * max_len + j] > max)
      {
        max = corpus[i * max_len + j];
      }
      if (corpus[i * max_len + j + 1] == -1)
      {
        break;
      }
    }
  }
  return max + 1;
}

void frequency_count(int *corpus, int corpus_size, int max_len, int **count_arr)
{
  for (size_t i = 0; i < corpus_size; i++)
  {
    for (size_t j = 0; j < max_len; j++)
    {
      if (corpus[i * max_len + j] != -1)
      {
        count_arr[i][corpus[i * max_len + j]]++;
      }
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

    // count the number of columns in the current line
    char *token = strtok(line, " ");
    while (token != NULL)
    {
      if (strcmp(token, "\n") != 0)
        num_columns++;
      token = strtok(NULL, " ");
    }

    // update the maximum number of columns seen so far
    if (num_columns > *num_cols)
    {
      *num_cols = num_columns;
    }

    // count non-empty lines
    if (num_columns > 0)
    {
      (*num_rows)++;
    }
  }

  // Allocate memory for the 2D array
  *array = (int *)malloc(*num_rows * *num_cols * sizeof(int));

  // Reset the file pointer to the beginning of the file
  fseek(fp, 0, SEEK_SET);

  // Read the integers in the file into the 2D array
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
  read_corpus("corpus_large.txt", &corpus, &corpus_size, &seq_max_len);
  printf("seq_max_len: %d, corpus size: %d\n", seq_max_len, corpus_size);

  // printf("input corpus\n");
  // for (size_t i = 0; i < corpus_size; i++)
  // {
  //   for (size_t j = 0; j < seq_max_len; j++)
  //   {
  //     printf("% 3d ", corpus[i * seq_max_len + j]);
  //   }
  //   printf("\n");
  // }

  vocab_size = find_vocab_size(corpus, corpus_size, seq_max_len);
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

  frequency_count(corpus, corpus_size, seq_max_len, counts);

  // printf("word counts in each doc:\n");
  // for (int i = 0; i < corpus_size; i++)
  // {
  //   for (size_t j = 0; j < vocab_size; j++)
  //   {
  //     printf("%d, ", counts[i][j]);
  //   }
  //   printf("\n");
  // }
  // printf("\n");

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

  // printf("tf:\n");
  // for (int i = 0; i < corpus_size; i++)
  // {
  //   for (size_t j = 0; j < vocab_size; j++)
  //   {
  //     printf("%.2f, ", tf_arr[i][j]);
  //   }
  //   printf("\n");
  // }
  // printf("\n");

  double *idf_arr = (double *)malloc(sizeof(double) * vocab_size);
  for (size_t wi = 0; wi < vocab_size; wi++)
  {
    int w_dc = 0 + smooth_idf;
    for (size_t di = 0; di < corpus_size; di++)
    {
      if (counts[di][wi])
        w_dc++;
    }

    idf_arr[wi] = log((double)corpus_size / w_dc) + 1;
  }

  // printf("idf:\n");
  // for (size_t wi = 0; wi < vocab_size; wi++)
  // {
  //   printf("%.3f, ", idf_arr[wi]);
  // }
  // printf("\n\n");

  double **tf_idf_arr = tf_arr;
  for (int di = 0; di < corpus_size; di++)
  {
    for (size_t wi = 0; wi < vocab_size; wi++)
    {
      tf_idf_arr[di][wi] *= idf_arr[wi];
    }
  }

  // printf("tf-idf:\n");
  // for (int i = 0; i < corpus_size; i++)
  // {
  //   for (size_t j = 0; j < vocab_size; j++)
  //   {
  //     printf("%.2f, ", tf_idf_arr[i][j]);
  //   }
  //   printf("\n");
  // }
  // printf("\n");
  return 0;
}
