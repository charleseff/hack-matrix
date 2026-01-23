# Colab TPU Quickstart

Get started with TPU training in 5 minutes using Google Colab (no signup or approval needed).

## Step 1: Open Notebook in Colab

### Option A: Direct Link (Easiest - Recommended)
Click this link to open directly in Colab:
```
https://colab.research.google.com/github/charleseff/hack-matrix/blob/main/python/notebooks/train_hackmatrix_tpu.ipynb
```

### Option B: From GitHub (Also Recommended)
1. Go to https://colab.research.google.com/
2. File → Open notebook → GitHub tab
3. Enter: `charleseff/hack-matrix`
4. Select `python/notebooks/train_hackmatrix_tpu.ipynb`

**Benefits of using GitHub:**
- ✅ Always get the latest version
- ✅ Easy to pull updates if code changes
- ✅ Can save a copy to your Drive with "Copy to Drive"
- ✅ Consistent with repository code

### Option C: Direct Upload (Fallback)
1. Download `python/notebooks/train_hackmatrix_tpu.ipynb` from the repo
2. Go to https://colab.research.google.com/
3. File → Upload notebook
4. Select the downloaded file

⚠️ **Note:** If you upload directly, you'll need to re-upload if the notebook is updated.

## Step 2: Enable TPU

1. Runtime → Change runtime type
2. Hardware accelerator → **TPU**
3. Click Save

## Step 3: Run Training

Click the play button on each cell in order:

1. ✅ **Cell 1**: Verify TPU detected
2. 📦 **Cell 2-3**: Clone repo & install dependencies (~2 minutes)
3. 🧪 **Cell 4**: Quick test (1K timesteps, ~10 seconds)
4. 🏃 **Cell 5**: Medium training (100K timesteps, ~1-2 minutes)
5. 🚀 **Cell 6**: Full training (10M timesteps, ~5-10 minutes)
6. 💾 **Cell 7**: Download checkpoints

## Expected Output

### Cell 1 (TPU Check):
```
JAX version: 0.4.x
Devices: [TpuDevice(id=0), TpuDevice(id=1), ..., TpuDevice(id=7)]
Device count: 8
Backend: tpu

✅ TPU detected! Ready to train.
```

### Cell 4 (Quick Test):
```
Device: TPU
Device count: 8
Backend: tpu

Training completed in 8.2s
Training complete! Total timesteps: 1,024
```

### Cell 6 (Full Training):
```
Training completed in 312.5s
Final metrics:
  mean_reward: 15.3
  total_loss: 0.024
  ...

Training complete! Total timesteps: 10,485,760
```

## Troubleshooting

### "TPU not detected"
- Make sure you selected TPU in Runtime settings
- Try Runtime → Restart runtime
- Check Colab status page: https://status.cloud.google.com/

### "Out of memory"
In the training cell, reduce these parameters:
```python
--num-envs 1024    # Instead of 2048
--hidden-dim 256   # Instead of 512
--num-steps 256    # Instead of 512
```

### "Session timed out"
- Colab free tier has 12-hour limit
- Download checkpoints regularly (Cell 7)
- Consider Colab Pro for longer sessions

### "Cannot find module 'hackmatrix'"
- Make sure Cell 2 (`git clone`) completed successfully
- Check you're in the right directory: `%cd hack-matrix/python`

## Next Steps

After successful TPU training:

1. **Download checkpoints** (Cell 7)
2. **Analyze results** - Check training metrics
3. **Compare with CPU** - TPU should be 50-100x faster
4. **Tune hyperparameters** - Adjust learning rate, network size, etc.
5. **Longer training** - Try 100M timesteps for better convergence

## Performance Benchmarks

Expected training times on Colab TPU (v2-8 or v3-8):

| Timesteps | Expected Time | Notes |
|-----------|---------------|-------|
| 1K        | 10 seconds    | Quick smoke test |
| 10K       | 15 seconds    | Verify compilation works |
| 100K      | 1-2 minutes   | See learning progress |
| 1M        | 5-10 minutes  | Decent policy emerges |
| 10M       | 5-10 minutes  | Good performance |
| 100M      | 30-60 minutes | Near-optimal (if Colab allows) |

Compare to CPU training times:
- 10M timesteps: ~3 hours on CPU vs ~5 minutes on TPU
- **~36x speedup!**

## Tips for Longer Training

### Colab Limits
- Free tier: 12 hours max per session
- Save checkpoints every 100 updates
- Download checkpoints to avoid losing progress

### Resume Training
If disconnected, you can resume by:
1. Rerun Cells 1-3 (setup)
2. Upload previous checkpoint
3. Modify training command to load checkpoint (feature coming soon)

### Monitoring
Watch the loss and reward metrics:
- `total_loss` should decrease
- `mean_reward` should increase
- `entropy` should decrease slowly (agent becomes more confident)

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify Colab TPU status: https://status.cloud.google.com/
3. Try restarting runtime: Runtime → Restart runtime
4. Report issues: https://github.com/charleseff/hack-matrix/issues
