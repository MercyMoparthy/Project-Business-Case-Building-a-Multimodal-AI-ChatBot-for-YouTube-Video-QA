import os

def save_chunks(chunks, out_file):
    with open(out_file, 'w') as f:
        for c in chunks:
            f.write(c + "\n\n")