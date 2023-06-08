# Image Hole Filling Project
## Usage
run `py main.py` for default behavior. running `py main.py -h` will trigger the following
```
main.py [-h] [image_path] [hole_path] [z] [eps] [connectivity] [algorithm]

Given an original image and a hole image, merges the two and fills the hole. The result is saved as merged.jpg. Use
flag -h for more information, or simply run py main.py for default behaviour.

positional arguments:
  image_path    Image file path (default=img.jpg).
  hole_path     Mask file path (same size as image, otherwise reshaped, default=mask.jpg).
  z             Power term for the distance metric (default=2).
  eps           Small float value used to avoid division by zero (default=0.01).
  connectivity  Either 4 or 8 (default=4).
  algorithm     Either original, or bonus1 (default=original).

optional arguments:
  -h, --help    show this help message and exit
```

## Examples
1. default behaviour:

 `py main.py`
 
![merged](https://user-images.githubusercontent.com/45313790/166115992-9ef2454c-7113-4360-8d4a-8ef736ce6b48.jpg)

2. bonus 1, setting 10 partitions (l=10)

 `py main.py img.jpg mask.jpg 2 0.01 8 bonus1`
 
![merged](https://user-images.githubusercontent.com/45313790/166116120-1e7bfe07-d008-4a5f-8723-b51de3d2bf7e.jpg)
