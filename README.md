# Classification-Models

### Hi, these are classification models that I tried and distilled models
### Training
```
python train.py --root /dataset --num_classes 2  --backbone unicom --freeze True --batch-size 64
```
### Distill Model

```
python distill_model.py --root /dataset --num_classes 2  --backbone unicom --freeze True --batch-size 64 --pretrained_teacher teacher.pt --pretrained_student student.pt
```
