# GroundedDINO-VL Quick Reference

## 🚀 Batch Inference (NEW!)

Process thousands of images locally without Label Studio:

```bash
# Basic usage
python -m groundeddino_vl.batch_inference \
    --input /data/datasets/bears \
    --output /data/results/bears \
    --config models/GroundingDINO_SwinB_cfg.py \
    --checkpoint checkpoints/groundingdino_swinb_cogcoor.pth \
    --visualize

# Custom classes for Night Vyper
python -m groundeddino_vl.batch_inference \
    --input /data/datasets/predators \
    --output /data/results/predators \
    --config models/GroundingDINO_SwinB_cfg.py \
    --checkpoint checkpoints/groundingdino_swinb_cogcoor.pth \
    --classes "bear,wolf,cougar,coyote,bobcat,fox" \
    --box-threshold 0.30 \
    --format labelstudio
```

**Output formats:** COCO JSON, YOLO, Label Studio JSON, CSV
**Performance:** ~10-16 img/s on RTX 4070 (22k images in ~25 min)
**See:** [BATCH_INFERENCE_QUICKSTART.txt](BATCH_INFERENCE_QUICKSTART.txt)

---

## 🎯 Label Studio Integration (What Was Fixed)

| Issue | Fix |
|-------|-----|
| ✅ Predictions show only after opening task | Fixed `from_name="label"` and `model_version` |
| ✅ Total Predictions = 0 | Added proper `model_version` at top level |
| ✅ Auto-annotation doesn't work | Fixed field name matching |
| ✅ Wrong image URLs | Added URL normalization (removes hostname/port) |
| ✅ Generic "a photo" prompt | Now uses 26 wildlife classes by default |

## 🚀 Quick Start

### 1. Restart Backend
```bash
cd /data/groundeddino-vl
docker-compose restart groundeddino-vl
```

### 2. Test Backend
```bash
./test_backend.sh
```

### 3. Configure Label Studio

**ML Backend URL:**
```
http://groundeddino-vl:9090
```

**Required Toggles (Settings → Machine Learning):**
- ✅ Get predictions when loading a task automatically
- ✅ Retrieve predictions when loading task automatically
- ✅ Predictions are displayed as initial annotations

## 📝 Label Studio Interface Config

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="bear" background="red"/>
    
  </RectangleLabels>
</View>
```

**Critical:** `name="label"` must match backend's `from_name="label"`

## 🧪 Test Commands

### Health Check
```bash
curl http://localhost:9090/health
```

### Test Prediction
```bash
curl -X POST http://localhost:9090/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "image": "/data/local-files/?d=/data/datasets/bear/bear_0001.jpg"
    }
  }'
```

### Expected Output Structure
```json
{
  "results": [
    {
      "result": [
        {
          "from_name": "label",
          "to_name": "image",
          "type": "rectanglelabels",
          "value": {
            "x": 10.5,
            "y": 15.2,
            "width": 25.8,
            "height": 30.1,
            "rotation": 0,
            "rectanglelabels": ["bear"],
            "score": 0.95
          }
        }
      ],
      "model_version": "1.0.0b1",
      "score": 0.95
    }
  ]
}
```

## 🔧 Customization

### Change Wildlife Classes

Edit [groundeddino_vl/ls_backend/server.py](groundeddino_vl/ls_backend/server.py#L66-L91):

```python
DEFAULT_WILDLIFE_CLASSES = [
    "bear", "deer", "coyote",  # Your custom list
]
```

### Adjust Detection Thresholds

Set environment variables in [docker-compose.yml](docker-compose.yml):

```yaml
environment:
  BOX_THRESHOLD: 0.30  # default: 0.25
  TEXT_THRESHOLD: 0.30  # default: 0.25
```

## 📊 Batch Auto-Annotation Workflow

1. **Select tasks** in Label Studio grid view (checkboxes)
2. **Actions → Retrieve Predictions** (loads predictions for all)
3. **Actions → Create Annotations from Predictions** (converts to annotations)
4. **Review each task** and submit

## 🐛 Troubleshooting

### Predictions not showing?
```bash
# Check backend logs
docker-compose logs -f groundeddino-vl

# Verify connection
curl http://localhost:9090/health
```

### Wrong classes detected?
- Backend defaults to 26 wildlife classes (no longer uses "a photo")
- Override per task with `"prompt": ["custom", "classes"]`

### Images not loading?
- Verify volume mount: `/data/datasets` in both containers
- Check file permissions
- Use absolute path: `/data/datasets/class/file.jpg`

## 📁 Key Files Modified

| File | What Changed |
|------|--------------|
| [server.py:188](groundeddino_vl/ls_backend/server.py#L188) | Added wildlife class list |
| [server.py:125](groundeddino_vl/ls_backend/server.py#L125) | Added URL normalization |
| [server.py:338](groundeddino_vl/ls_backend/server.py#L338) | Updated /predict to normalize URLs |
| [inference_engine.py:188](groundeddino_vl/ls_backend/inference_engine.py#L188) | Changed from_name to "label" |
| [utils.py:363](groundeddino_vl/ls_backend/utils.py#L363) | Changed from_name to "label" |

## 🎉 Success Checklist

- [ ] Backend health check passes (`./test_backend.sh`)
- [ ] Label Studio shows ML backend connected (green checkmark)
- [ ] Opening a task automatically loads predictions
- [ ] Predictions show correct wildlife class labels
- [ ] "Total Predictions per Task" shows > 0
- [ ] "Create Annotations from Predictions" works for batch
- [ ] Image URLs work (no 404 errors)

## 📚 Documentation

- **Full Guide:** [LABELSTUDIO_FIX_GUIDE.md](LABELSTUDIO_FIX_GUIDE.md)
- **Test Suite:** [test_backend.sh](test_backend.sh)
- **Docker Setup:** [docker-compose.yml](docker-compose.yml)

---

**Need help?** Check logs: `docker-compose logs -f groundeddino-vl`
