"""This code is used to delete analysis result folder
    from previous tests
"""
import os
import shutil

MODEL_FOLDER = os.path.join(os.getcwd(), 'test_model')

if os.path.exists(MODEL_FOLDER):
    shutil.rmtree(MODEL_FOLDER)