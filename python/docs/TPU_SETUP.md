# TPU Training Setup for HackMatrix

## Prerequisites

1. ✅ Google Cloud project created
2. ✅ Cloud TPU API enabled
3. ✅ Project number submitted to TRC
4. ✅ Billing account set up (for storage/networking, not TPUs)
5. ⏳ Waiting for TRC confirmation email

## TPU VM Setup

### 1. Install Google Cloud SDK

```bash
# On your local machine (not dev container)
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init
```

### 2. Create TPU VM

Once TRC approves your project, create a TPU VM:

```bash
# Set project
gcloud config set project YOUR_PROJECT_ID

# Create TPU v3-8 (8 cores, good for small/medium models)
gcloud compute tpus tpu-vm create hackmatrix-tpu \
  --zone=us-central2-b \
  --accelerator-type=v3-8 \
  --version=tpu-ubuntu2204-base

# Or TPU v4-8 (newer, faster if available)
gcloud compute tpus tpu-vm create hackmatrix-tpu \
  --zone=us-central2-b \
  --accelerator-type=v4-8 \
  --version=tpu-ubuntu2204-base
```

### 3. SSH into TPU VM

```bash
gcloud compute tpus tpu-vm ssh hackmatrix-tpu --zone=us-central2-b
```

### 4. Set Up Environment on TPU VM

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Python 3.10
sudo apt-get install -y python3.10 python3.10-venv python3-pip git

# Clone your repo
git clone https://github.com/YOUR_USERNAME/hack-matrix.git
cd hack-matrix/python

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install JAX for TPU
pip install -U pip
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libjax_releases.html

# Install other dependencies
pip install -r requirements.txt
```

### 5. Verify TPU Detection

```bash
source venv/bin/activate
python -c "import jax; print('Devices:', jax.devices()); print('TPU cores:', len(jax.devices()))"
```

Expected output:
```
Devices: [TpuDevice(id=0), TpuDevice(id=1), ..., TpuDevice(id=7)]
TPU cores: 8
```

### 6. Build Swift Binary (for headless mode)

The JAX environment doesn't need the Swift binary, but if you want to test the Swift gym env:

```bash
# Install Swift (on Ubuntu)
wget https://download.swift.org/swift-5.9.2-release/ubuntu2204/swift-5.9.2-RELEASE/swift-5.9.2-RELEASE-ubuntu22.04.tar.gz
tar xzf swift-5.9.2-RELEASE-ubuntu22.04.tar.gz
sudo mv swift-5.9.2-RELEASE-ubuntu22.04 /usr/share/swift
echo 'export PATH=/usr/share/swift/usr/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Build
cd /path/to/hack-matrix
swift build -c release
```

## Running Training on TPU

### Quick Test (1K timesteps)

```bash
source venv/bin/activate
python scripts/train_purejaxrl.py \
  --num-envs 256 \
  --num-steps 128 \
  --total-timesteps 1000 \
  --seed 42
```

### Full Training (10M timesteps)

```bash
source venv/bin/activate

# Use screen or tmux for long training
screen -S training

python scripts/train_purejaxrl.py \
  --num-envs 2048 \
  --num-steps 256 \
  --total-timesteps 10000000 \
  --save-interval 100 \
  --log-interval 10 \
  --checkpoint-dir checkpoints/tpu_run_1 \
  --seed 42

# Detach from screen: Ctrl+A, D
# Reattach: screen -r training
```

### Optimal TPU Hyperparameters

For TPU v3-8 or v4-8:

```bash
python scripts/train_purejaxrl.py \
  --num-envs 2048 \
  --num-steps 256 \
  --total-timesteps 100000000 \
  --learning-rate 0.0003 \
  --num-minibatches 8 \
  --update-epochs 4 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --clip-eps 0.2 \
  --ent-coef 0.01 \
  --vf-coef 0.5 \
  --hidden-dim 512 \
  --num-layers 3 \
  --save-interval 100 \
  --checkpoint-dir checkpoints/tpu_production
```

## Monitoring Training

### Check Progress

```bash
# View recent logs
tail -f training.log

# Check GPU/TPU utilization (if monitoring tools available)
watch -n 1 nvidia-smi  # For GPU
# For TPU, use Cloud Console monitoring
```

### Download Checkpoints

From your local machine:

```bash
# Download latest checkpoint
gcloud compute tpus tpu-vm scp \
  hackmatrix-tpu:~/hack-matrix/python/checkpoints/final_params.npz \
  ./local_checkpoints/ \
  --zone=us-central2-b
```

## Cost Management

### Free Tier Limits

- **TPU time**: Free for 30 days (TRC program)
- **Storage**: $300 free credit should cover checkpoints
- **Networking**: Minimal cost for SSH/data transfer

### Stop TPU When Not Training

```bash
# Stop (preserves disk, stops compute charges)
gcloud compute tpus tpu-vm stop hackmatrix-tpu --zone=us-central2-b

# Start again
gcloud compute tpus tpu-vm start hackmatrix-tpu --zone=us-central2-b

# Delete when done (frees all resources)
gcloud compute tpus tpu-vm delete hackmatrix-tpu --zone=us-central2-b
```

### Monitor Costs

- Dashboard: https://console.cloud.google.com/billing
- Set up budget alerts to avoid surprises

## Troubleshooting

### JAX Can't Find TPU

```bash
# Check TPU is running
gcloud compute tpus tpu-vm list --zone=us-central2-b

# Reinstall JAX for TPU
pip uninstall jax jaxlib -y
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libjax_releases.html
```

### Out of Memory

- Reduce `--num-envs` (try 1024 instead of 2048)
- Reduce `--hidden-dim` (try 256 instead of 512)
- Reduce `--num-steps` (try 128 instead of 256)

### Slow Compilation

- First run compiles everything (5-30 seconds normal)
- Subsequent updates should be fast (<1s)
- If recompiling every step, check for dynamic shapes

## Next Steps After TRC Approval

1. Create TPU VM (see above)
2. Run quick test (1K timesteps)
3. Run medium test (100K timesteps, ~5-10 minutes)
4. Run full training (10M+ timesteps, hours/days)
5. Download checkpoints regularly
6. Monitor costs in billing dashboard

## Performance Expectations

### Throughput (approximate)

- **CPU**: ~1K steps/sec
- **GPU (V100)**: ~10K steps/sec
- **TPU v3-8**: ~50K steps/sec
- **TPU v4-8**: ~100K steps/sec

### Training Time Estimates

For 10M timesteps:
- **CPU**: ~3 hours
- **GPU**: ~15 minutes
- **TPU v3-8**: ~3 minutes
- **TPU v4-8**: ~1.5 minutes

For 100M timesteps:
- **CPU**: ~30 hours
- **GPU**: ~2.5 hours
- **TPU v3-8**: ~30 minutes
- **TPU v4-8**: ~15 minutes
