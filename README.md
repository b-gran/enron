# Exploratory analysis of the attachments in the Enron email corpus

![/Users/bill/experiments/enron-analysis/enron-all-blurred.png]

### Usage

0. Figure out the requirements and `pip install` them. Sorry 😅
1. Download the [full dataset with attachments (Internet Archive)](https://archive.org/details/edrm.enron.email.data.set.v2.xml).
2. `python data.py --enron-root $path_to_download --media-dir $image_df_output_path --hidden`
3. `python viz.py --input $image_df_output_path --output $big_image_output_path`
4. [for tagging emails with OpenAI] `python tags.py -i $path_to_kaggle_text_dataframe_joblib` (this costs ~$50 as of September 2024)
5. Check of `results.ipynb` for some hints on how to work with the tagged text data

### References
* [Full dataset with attachments (Internet Archive)](https://archive.org/details/edrm.enron.email.data.set.v2.xml)
* [Cleaned 2015 revision of the email text](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset)