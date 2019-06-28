"""
This script is used to add new tables to accumulated tables

$ python3 add_to_table.py main_table.csv small_table1.csv small_table2.csv 

"""
import pandas as pd
import numpy as np
import os
import sys

def add_to_frame(frame, fname):
    new_frame = pd.read_csv(fname, delimiter="!")
    if frame is None:
        return new_frame
    new_frame.reindex(frame.columns, axis=1)
    mismatch = False
    for col1, col2 in zip(frame.axes[1], new_frame.axes[1]):
        if not mismatch and col1 != col2:
            print("Name:", fname)
            mismatch = True
            s1 = set(frame.columns)
            s2 = set(new_frame.columns)
            print("Frame:", s1-s2)
            print()
            print("New:", s2-s1)
            print()
            print()
            print()
    frame = frame.append(new_frame)
    return frame

if __name__ == "__main__":
    assert len(sys.argv) >= 3
    main_frame = pd.read_csv(sys.argv[1], delimiter="!")
    main_frame.to_csv(sys.argv[1]+".backup", header=True, index=False, sep="!")
    for arg in sys.argv[2:]:
        main_frame = add_to_frame(main_frame, arg)
    main_frame.to_csv(sys.argv[1], header=True, index=False, sep="!")
