#!/bin/bash
echo "Welcom to JDDC 2020"
python3 online_test_data_preprocess.py
python3 online_test_inference.py --checkpoint=/xxx/mhred/chk/1.pkl
python3 online_test_data_postprocess.py
echo "Done!"