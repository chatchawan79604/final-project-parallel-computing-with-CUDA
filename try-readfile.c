#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define MAX_LINE_LENGTH 1024

void count_columns_and_lines(char *filename, int **array, int *num_rows, int *num_cols)
{
  FILE *fp;
  size_t line_size = MAX_LINE_LENGTH;
  char line[line_size];
  int num_columns, num_lines;
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
  int num_rows, num_cols;
  int *array;
  count_columns_and_lines("corpus_1.txt", &array, &num_rows, &num_cols);
  printf("Array dimensions: %d x %d\n", num_rows, num_cols);

  // Access the elements of the array like this:
  for (int i = 0; i < num_rows; i++)
  {
    for (int j = 0; j < num_cols; j++)
    {
      printf("% 3d ", array[i * num_cols + j]);
    }
    printf("\n");
  }

  free(array); // Free the memory allocated by malloc
  return 0;
}
