#include <stdio.h>
#include <string.h>

int find_word(char *word, char *words[], int n) {
    int i;
    for (i = 0; i < n; i++) {
        if (strcmp(word, words[i]) == 0) {
            return i;
        }
    }
    return -1;
}

int count_word(char *word, char *words[], int n) {
    int i;
    int count = 0;
    for (i = 0; i < n; i++) {
        if (strcmp(word, words[i]) == 0) {
            count++;
        }
    }
    return count;
}

int main() {

    char *words[] = {"the", "the", "brown", "fox", "the", "over", "the", "lazy", "dog"};
    int n = 9;

    for (int i = 0; i < n; i++) {
        int count = 0;
        count = count_word(words[i], words, n);
        printf("%s %d\n", words[i], count);
    }

    return 0;
}