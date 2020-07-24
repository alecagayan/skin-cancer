Possible title

Skin cancer diagnosis using deep learning

Skin cancer diagnosis using deep neural networks

Data Sources
[1] Noel Codella, Veronica Rotemberg, Philipp Tschandl, M. Emre Celebi, Stephen Dusza, David Gutman, Brian Helba, Aadi Kalloo, Konstantinos Liopyris, Michael Marchetti, Harald Kittler, Allan Halpern: “Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)”, 2018; https://arxiv.org/abs/1902.03368
[2] Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci. Data 5, 180161 doi:10.1038/sdata.2018.161 (2018).

Chollet, Francois. “The Keras Blog.” The Keras Blog ATOM, https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html.

“ISIC Archive.” ISIC Archive, https://www.isic-archive.com/.

Hagan, Martin T., et al. Neural Network Design. s. n., 2016.

LeCun, Y., Bengio, Y. & Hinton, G. Deep learning. Nature 521, 436–444 (2015)


HAM Prep

2020-01-11 23:11:20.468663: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cudart64_100.dll
     lesion_id      image_id   dx dx_type   age   sex localization
0  HAM_0000118  ISIC_0027419  bkl   histo  80.0  male        scalp
1  HAM_0000118  ISIC_0025030  bkl   histo  80.0  male        scalp
2  HAM_0002730  ISIC_0026769  bkl   histo  80.0  male        scalp
3  HAM_0002730  ISIC_0025661  bkl   histo  80.0  male        scalp
4  HAM_0001466  ISIC_0031633  bkl   histo  75.0  male          ear
     lesion_id  image_id  dx  dx_type  age  sex  localization
0  HAM_0000001         1   1        1    1    1             1
1  HAM_0000003         1   1        1    1    1             1
2  HAM_0000004         1   1        1    1    1             1
3  HAM_0000007         1   1        1    1    1             1
4  HAM_0000008         1   1        1    1    1             1
     lesion_id      image_id   dx dx_type   age   sex localization      duplicates
0  HAM_0000118  ISIC_0027419  bkl   histo  80.0  male        scalp  has_duplicates
1  HAM_0000118  ISIC_0025030  bkl   histo  80.0  male        scalp  has_duplicates
2  HAM_0002730  ISIC_0026769  bkl   histo  80.0  male        scalp  has_duplicates
3  HAM_0002730  ISIC_0025661  bkl   histo  80.0  male        scalp  has_duplicates
4  HAM_0001466  ISIC_0031633  bkl   histo  75.0  male          ear  has_duplicates
no_duplicates     5514
has_duplicates    4501
Name: duplicates, dtype: int64
(5514, 8)
(938, 8)
9077
938
nv       5954
mel      1074
bkl      1024
bcc       484
akiec     301
vasc      131
df        109
Name: dx, dtype: int64
nv       751
bkl       75
mel       39
bcc       30
akiec     26
vasc      11
df         6
Name: dx, dtype: int64