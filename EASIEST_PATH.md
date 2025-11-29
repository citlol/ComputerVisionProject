# EASIEST PATH TO DEMO  for MacBook Air

## Setup
- **MacBook**: Apple Silicon (M1/M2/M3) with MPS GPU ✓
- **Time needed**: 2-3 hours total
- **Current images**: 9 training + 2 validation (need more!)

## Three Simple Options

### Option 1: Download CelebA Faces (RECOMMENDED) 
**Best for impressive demo results**

1. **Go to Kaggle**: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
   - Click "Download" (need free Kaggle account)
   - Download just the `img_align_celeba.zip` (1.3 GB)

2. **Extract and copy**:
   ```bash
   cd ~/Downloads
   unzip img_align_celeba.zip

   cd /Users/cittrejodelrio/Downloads/homework1_programming/Project/image_inpainting

   # Copy first 80 images to train
   ls ~/Downloads/img_align_celeba/*.jpg | head -80 | xargs -I {} cp {} data/train/

   # Copy next 15 to val
   ls ~/Downloads/img_align_celeba/*.jpg | head -95 | tail -15 | xargs -I {} cp {} data/val/

   # Copy next 15 to test
   ls ~/Downloads/img_align_celeba/*.jpg | head -110 | tail -15 | xargs -I {} cp {} data/test/

   # Verify
   echo "Training: $(ls data/train | wc -l)"
   echo "Validation: $(ls data/val | wc -l)"
   echo "Test: $(ls data/test | wc -l)"
   ```

3. **Train** (takes ~2 hours):
   ```bash
   ./train_macbook.sh
   ```

**Total time**: 30 min download + 2 hours training = **2.5 hours**

---

### Option 2: Use Your Own Photos (FASTEST) 🏃
**Best if you have photos available**

1. **Collect 50-100 photos** from:
   - Your Photos app
   - iPhone/Android photos
   - Any personal photos (faces, landscapes, objects)

2. **Copy to project**:
   ```bash
   cd /Users/cittrejodelrio/Downloads/homework1_programming/Project/image_inpainting

   # Copy your photos (adjust path)
   cp ~/Pictures/MyPhotos/*.jpg data/train/
   # Should have 50-80 photos

   # Move last 15 to validation
   ls data/train/*.jpg | tail -15 | xargs -I {} mv {} data/val/

   # Move last 10 to test
   ls data/train/*.jpg | tail -10 | xargs -I {} mv {} data/test/
   ```

3. **Train**:
   ```bash
   ./train_macbook.sh
   ```

**Total time**: 15 min prep + 2 hours training = **2.25 hours**

---

### Option 3: Minimal Demo (Just Show Interface) 🎯
**If you're short on time - NO real training needed**

Skip training, just demonstrate the interface and architecture:

1. **Run demo with untrained model**:
   ```bash
   ./run_demo.sh
   # Results will be blurry but shows the interface
   ```

2. **Use architecture visualization**:
   ```bash
   python demo_no_training.py
   # Shows how the model works
   ```

3. **In your presentation, explain**:
   - "This shows the interface and architecture"
   - "With trained model (2-3 hours on MacBook Air), results would be realistic"
   - "Model uses Partial Convolutions with 33M parameters"
   - Show architecture_demo.png

**Total time**: **30 minutes** (no training)

**Good for**: Tight deadlines, focus on methodology over results

---


**3+ hours**: Use **Option 1** (CelebA faces)
- Download takes 30 minutes
- Training takes 2 hours
- Best visual results for demo

**Personal photos**: Use **Option 2**
- Fastest to prepare
- Still gets good results

**< 2 hours**: Use **Option 3**
- Shows interface and architecture
- Be upfront: "Demo of system, training requires 2-3 hours"

---

## QUICK START 

```bash
# 1. Download CelebA (30 min)
# Visit: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset

# 2. Extract and copy (5 min)
cd ~/Downloads
unzip img_align_celeba.zip

cd /Users/cittrejodelrio/Downloads/homework1_programming/Project/image_inpainting

ls ~/Downloads/img_align_celeba/*.jpg | head -80 | xargs -I {} cp {} data/train/
ls ~/Downloads/img_align_celeba/*.jpg | head -95 | tail -15 | xargs -I {} cp {} data/val/
ls ~/Downloads/img_align_celeba/*.jpg | head -110 | tail -15 | xargs -I {} cp {} data/test/

# 3. Verify
echo "Train: $(ls data/train | wc -l), Val: $(ls data/val | wc -l), Test: $(ls data/test | wc -l)"

# 4. Train (2 hours)
./train_macbook.sh

# 5. Test
python evaluate.py --test_dir data/test --checkpoint checkpoints/best.pth

# 6. Demo
./run_demo.sh


```

---

## Expected Training Time (MacBook Air M1/M2)

| Images | Batch Size | Epochs | Time per Epoch | Total Time |
|--------|------------|--------|----------------|------------|
| 50     | 2          | 25     | 4 min          | ~1.5 hrs   |
| 80     | 2          | 25     | 5 min          | ~2 hrs     |
| 100    | 2          | 25     | 6 min          | ~2.5 hrs   |

---



---


