# CelebA Training Checklist for MacBook Air

## 📋 Complete Timeline (2.5-3 hours total)

### Phase 1: Download Dataset (~30 minutes)

- [ ] Go to https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
- [ ] Create/login to Kaggle account
- [ ] Click "Download" button
- [ ] Wait for `img_align_celeba.zip` (1.3 GB) to download to ~/Downloads

**⏰ While waiting:** Start preparing your presentation slides!

---

### Phase 2: Setup Dataset (~5 minutes)

Once download completes:

```bash
cd /Users/cittrejodelrio/Downloads/homework1_programming/Project/image_inpainting

# Run setup script
./setup_celeba.sh
```

**Expected output:**
```
✓ Found img_align_celeba.zip in Downloads
✓ Extraction complete
✓ Dataset Setup Complete!

Final dataset:
  Training:   80 images
  Validation: 15 images
  Test:       15 images
```

- [ ] Setup script completed successfully
- [ ] Verified image counts (80/15/15)

---

### Phase 3: Start Training (~2 hours)

```bash
# Start training
./train_macbook.sh
```

**What you'll see:**
```
Found 80 training images
Found 15 validation images
Starting optimized training for MacBook Air...
Settings:
  - Device: MPS (Apple Silicon GPU)
  - Batch size: 2
  - Image size: 256x256
  - Epochs: 25

Using device: mps
Loading datasets...
Creating model...
Starting training from epoch 0...
Epoch 0: 100%|████████| [loss, psnr, ssim values]
```

**Training Progress:**
- [ ] Epoch 0 started (first epoch is slowest ~8-10 min)
- [ ] Epoch 1-5 completed (~5 min each)
- [ ] Epoch 10 reached - check `results/epoch_10.png` for progress
- [ ] Epoch 15 reached - inpainting should look better
- [ ] Epoch 20 reached - results should be good
- [ ] Epoch 25 completed - training done!

**⏰ While training:**
- Your Mac will get warm (normal!)
- You can use it lightly (web browsing, writing)
- Avoid heavy tasks (video editing, gaming)
- Check `results/epoch_XX.png` files to see progress

**Training Timeline:**
```
Epoch 1-5:   0:00 - 0:25 (25 minutes)
Epoch 6-10:  0:25 - 0:50 (25 minutes)
Epoch 11-15: 0:50 - 1:15 (25 minutes)
Epoch 16-20: 1:15 - 1:40 (25 minutes)
Epoch 21-25: 1:40 - 2:05 (25 minutes)
Total: ~2 hours
```

**Early Stopping Option:**
- After epoch 15-20, check if results are good enough
- If `results/epoch_15.png` looks decent, you can stop (Ctrl+C)
- Use `checkpoints/epoch_15.pth` or `checkpoints/latest.pth`

---

### Phase 4: Evaluate Model (~5 minutes)

After training completes:

```bash
# Evaluate on test set
python evaluate.py \
    --test_dir data/test \
    --checkpoint checkpoints/best.pth \
    --mask_type center
```

**Expected output:**
```
==================================================
Evaluation Results:
==================================================
Full Image Metrics:
  PSNR: 28.5 dB
  SSIM: 0.91

Hole Region Metrics:
  PSNR: 26.2 dB
  SSIM: 0.88
==================================================
```

**Target Metrics:**
- [ ] PSNR > 25 dB (good), > 28 dB (excellent)
- [ ] SSIM > 0.85 (good), > 0.90 (excellent)

**Save these numbers for your presentation!**

---

### Phase 5: Test Demo Interface (~5 minutes)

```bash
# Start web demo
./run_demo.sh
```

**Opens at:** http://localhost:7860

**Test the demo:**
- [ ] Upload an image from `data/test/`
- [ ] Try "Center" mask → Click "Inpaint"
- [ ] Results look good!
- [ ] Try "Irregular" mask → Click "Inpaint"
- [ ] Results look good!
- [ ] Take screenshots for presentation

---

### Phase 6: Record Demo Video (~30 minutes)

**Prepare:**
- [ ] Close unnecessary apps
- [ ] Clean up desktop/browser tabs
- [ ] Have demo script ready
- [ ] Test audio (if recording voice)

**Record with QuickTime:**
1. Open QuickTime Player
2. File → New Screen Recording
3. Click Options → Choose microphone (if doing voiceover)
4. Click record button → Select full screen or area
5. Start your demo!

**Demo Script (5-7 minutes):**

**0:00 - 1:00** Introduction
- "I'm presenting image inpainting using Partial Convolutions"
- "Trained on CelebA face dataset with 80 images for 25 epochs"
- "Running on MacBook Air M1/M2 with MPS acceleration"

**1:00 - 2:00** Architecture Overview
- Show `demo_images/architecture_demo.png`
- Explain U-Net with Partial Convolutions
- Mention 33M parameters

**2:00 - 4:00** Live Demo - Center Mask
- Open web interface
- Upload test image
- Select "Automatic Mask" → "Center"
- Click "Inpaint"
- Show result: "Model successfully reconstructs facial features"

**4:00 - 5:30** Live Demo - Irregular Mask
- Upload another test image
- Select "Irregular" mask
- Click "Inpaint"
- Explain: "Handles complex, irregular holes"

**5:30 - 6:30** Metrics & Results
- Show evaluation metrics
- "PSNR: [your value] dB"
- "SSIM: [your value] - exceeds target of 0.9"
- Show side-by-side comparisons

**6:30 - 7:00** Conclusion
- "Successfully implemented Partial Convolution inpainting"
- "Achieved high quality results on MacBook Air"
- "Thank you!"

- [ ] Demo video recorded
- [ ] Video saved and tested

---

## 📊 Expected Results

After training on CelebA:

| Metric | Expected Range | Notes |
|--------|----------------|-------|
| PSNR (full) | 26-32 dB | Measures reconstruction accuracy |
| SSIM (full) | 0.88-0.94 | Structural similarity (target >0.9) |
| PSNR (hole) | 24-28 dB | Just in hole regions |
| SSIM (hole) | 0.84-0.90 | Just in hole regions |

**Visual Quality:**
- Center masks: Excellent (faces reconstruct well)
- Irregular masks: Good to Very Good
- Face features: Eyes, nose, mouth preserved
- Textures: Skin tone and details maintained

---

## 🔧 Troubleshooting

### Training stops or crashes
**Issue:** "MPS backend out of memory"
**Fix:**
```bash
# Reduce batch size
python train.py --train_dir data/train --val_dir data/val --batch_size 1 --epochs 25
```

### Mac gets very hot
**Solution:**
- Normal during training
- Ensure good ventilation
- Consider using laptop stand
- Optional: reduce workload, train overnight

### Training seems stuck
**Check:**
- Activity Monitor → Python process using CPU/GPU?
- First epoch takes 8-10 minutes (loading data)
- Subsequent epochs: ~5 minutes each
- Look for progress bar updates

### Results not improving after epoch 10
**Options:**
1. Continue training - may improve by epoch 20-25
2. Try smaller learning rate:
   ```bash
   python train.py --lr 0.0001 --train_dir data/train --val_dir data/val
   ```

---

## ✅ Final Checklist Before Demo

- [ ] Training completed (25 epochs or stopped early)
- [ ] Best checkpoint saved: `checkpoints/best.pth`
- [ ] Evaluation metrics recorded
- [ ] Demo interface tested and working
- [ ] Demo video recorded
- [ ] Presentation slides prepared with:
  - [ ] Problem statement
  - [ ] Architecture diagram
  - [ ] Training details (CelebA, 80 images, 25 epochs)
  - [ ] Results and metrics
  - [ ] Demo video embedded
  - [ ] Conclusion

---

## 📝 Key Information for Presentation

**Dataset:** CelebA (Celebrity Faces)
**Training Set:** 80 face images
**Validation Set:** 15 images
**Test Set:** 15 images
**Total Images:** 110 selected from 200k+ dataset

**Model:** Partial Convolution U-Net
**Parameters:** 33 million
**Architecture:** Encoder-decoder with skip connections

**Training:**
**Hardware:** MacBook Air M1/M2 with MPS GPU
**Epochs:** 25
**Batch Size:** 2
**Image Size:** 256×256
**Training Time:** ~2 hours
**Optimizer:** Adam (lr=0.0002)

**Loss Functions:**
- L1 Loss (valid + hole regions)
- Perceptual Loss (VGG16 features)
- Style Loss (Gram matrices)
- Total Variation Loss

**Evaluation Metrics:**
- PSNR: [your value] dB
- SSIM: [your value]
- Visual Quality: Excellent

---

## 🎯 Success Criteria

**Minimum for good demo:**
- ✓ PSNR > 25 dB
- ✓ SSIM > 0.85
- ✓ Visually convincing face reconstruction

**Excellent demo:**
- ✓ PSNR > 28 dB
- ✓ SSIM > 0.90
- ✓ Natural-looking inpainted faces

---

**CURRENT STATUS:** Ready to download CelebA dataset
**NEXT STEP:** Download from Kaggle, then run `./setup_celeba.sh`

Good luck! 🚀
