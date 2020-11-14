from CosineMeasure import CosineMeasure
#!/usr/bin/env python3
"""
Module Docstring
"""

__author__ = "Alfarady Raja Ghanie Hamid Jauhar"
__version__ = "1.0.0"
__license__ = "MIT"

DOCS = ["Gagal panen banyak yang terjadi",
        "Panen raya banyak dilaksanakan",
        "Jalan raya sering terjadi kecelakaan",
        "Petani gagal tanam karena mengalami panen yang gagal"]

def main():
    print("\033[95mCosine Measure\033[0m")
    print("\033[94mDocuments: ")
    for i in range(len(DOCS)):
        print(f'D{i+1}: {DOCS[i]}')
    print("\033[0m")
    var = input("\033[96mPlease input query: \033[0m")
    measure = CosineMeasure(query=var, documents=DOCS)
    measure.find_tfidf(is_query=0)
    measure.find_tfidf(is_query=1)
    measure.print_cosin_sim()


if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()