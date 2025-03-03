# VGG Feature Extraction and Visualization

## Install packages

- Python 3.12.8

```bash
pip install uv
uv pip install -r requirements.txt
```

## Replace VGG model

- Line 14 in main.py

```python
# VGG16
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

# VGG19
model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
```

## Run the app

```bash
python main.py
```
