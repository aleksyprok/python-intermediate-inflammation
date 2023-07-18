class Book:
    "Stores title and author."
    def __init__(self, title, author):
        self.title = title
        self.author = author

    def __str__(self):
        return self.title + " by " + self.author


import multiprocessing
import os

print("os.cpu_count() =", os.cpu_count())

def square(n):
    return n * n

if __name__ == "__main__":
    num_processors = 1  # Change this to set number of processors
    with multiprocessing.Pool(processes=num_processors) as pool:
        numbers = range(1000)
        squares = pool.map(square, numbers)
        print(f'Sum of squares: {sum(squares)}')

