import pytest
import os
import glob

def test_model():
    os.system('python scripts/main.py -config=configs/debugger.cellbox.json')
    files = glob.glob('results/Debugging_*/b11_000/3_best.W*')
    assert len(files)==1

if __name__ == '__main__':

    pytest.main(args=['-sv', os.path.abspath(__file__)])
