#!/usr/bin/env python3
def matrix_transpose(matrix):
              return [[matrix[row][col] for row in range(len(matrix))]
                                  for col in range(len(matrix[0]))]
