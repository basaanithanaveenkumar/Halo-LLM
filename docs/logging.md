# Experiment Logging

HaloDiffusionLLM supports multiple experiment tracking backends for comprehensive monitoring and visualization of training runs.

## Supported Backends

### 1. TensorBoard (Local)
TensorBoard provides local visualization of training metrics, model graphs, and hyperparameters.

**Features:**
- Real-time metric visualization
- Hyperparameter tracking
- Model graph visualization
- Histogram logging
- Image logging

**Configuration:**
```yaml
logging:
  tensorboard: true
  tensorboard_dir: data/tb
```

**Usage:**
```bash
# Start training with TensorBoard
ha-llm-train --config configs/diffusion/small.yaml

# View logs in browser
tensorboard --logdir data/tb
```

### 2. Weights & Biases (Cloud)
W&B provides cloud-based experiment tracking with advanced features like hyperparameter sweeps, model versioning, and team collaboration.

**Features:**
- Cloud-based experiment tracking
- Hyperparameter sweeps
- Model versioning
- Team collaboration
- Advanced visualizations
- Model gradient tracking

**Configuration:**
```yaml
logging:
  wandb: true
  wandb_project: "halodiffusion-llm"
  wandb_entity: "your-username-or-team"
  wandb_run_name: null  # Auto-generated if null
  wandb_tags: ["diffusion", "small"]
  wandb_watch_model: true  # Track gradients
  wandb_log_freq: 100  # Log frequency for gradients
```

**Setup:**
```bash
# Install W&B
pip install wandb

# Login (one-time setup)
wandb login

# Start training with W&B
ha-llm-train --config configs/diffusion/small.yaml
```

## Configuration Options

### TensorBoard Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tensorboard` | bool | `true` | Enable TensorBoard logging |
| `tensorboard_dir` | str | `"data/tb"` | Directory for TensorBoard logs |

### Weights & Biases Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `wandb` | bool | `false` | Enable W&B logging |
| `wandb_project` | str | `null` | W&B project name (required if enabled) |
| `wandb_entity` | str | `null` | W&B username or team name |
| `wandb_run_name` | str | `null` | Custom run name (auto-generated if null) |
| `wandb_tags` | list[str] | `null` | Tags for organizing runs |
| `wandb_watch_model` | bool | `false` | Track model gradients and parameters |
| `wandb_log_freq` | int | `100` | Frequency for gradient logging |

## Example Configurations

### TensorBoard Only (Default)
```yaml
logging:
  level: INFO
  log_file: data/logs/ha_llm.log
  tensorboard: true
  tensorboard_dir: data/tb
  wandb: false
```

### W&B Only
```yaml
logging:
  level: INFO
  log_file: data/logs/ha_llm.log
  tensorboard: false
  wandb: true
  wandb_project: "halodiffusion-llm"
  wandb_entity: "my-team"
  wandb_tags: ["experiment", "v1"]
```

### Both TensorBoard and W&B
```yaml
logging:
  level: INFO
  log_file: data/logs/ha_llm.log
  tensorboard: true
  tensorboard_dir: data/tb
  wandb: true
  wandb_project: "halodiffusion-llm"
  wandb_entity: "my-team"
  wandb_run_name: "diffusion-small-exp1"
  wandb_tags: ["diffusion", "small", "baseline"]
  wandb_watch_model: true
  wandb_log_freq: 100
```

### W&B with Gradient Tracking
```yaml
logging:
  level: INFO
  tensorboard: true
  tensorboard_dir: data/tb
  wandb: true
  wandb_project: "halodiffusion-llm"
  wandb_watch_model: true  # Enable gradient tracking
  wandb_log_freq: 50  # Log gradients every 50 steps
```

## Logged Metrics

Both backends automatically log:

### Training Metrics
- `train/loss` - Training loss per step
- `train/lr` - Learning rate per step
- `train/grad_norm` - Gradient norm per step
- `train/epoch` - Current epoch
- `train/epoch_loss` - Average loss per epoch
- `train/final_loss` - Final training loss

### Evaluation Metrics
- `eval/loss` - Validation loss
- `eval/perplexity` - Model perplexity
- `eval/bits_per_token` - Bits per token
- `eval/token_accuracy` - Token-level accuracy
- `eval/masked_accuracy` - Masked token accuracy

### Model Metrics
- `model/n_params` - Total number of parameters

### Hyperparameters
- `variant` - Model variant
- `lr` - Learning rate
- `batch_size` - Batch size
- `d_model` - Model dimension
- `n_layers` - Number of layers
- `n_heads` - Number of attention heads
- `max_length` - Maximum sequence length
- `epochs` - Total epochs
- `steps` - Total steps
- `loss_type` - Loss function type
- `parallel_strategy` - Data parallelism strategy
- `world_size` - Number of distributed processes

## CLI Usage

### Override W&B Settings via CLI
```bash
# Enable W&B for a single run
ha-llm-train --config configs/diffusion/small.yaml \
  --wandb-project "my-project" \
  --wandb-entity "my-team"

# Disable logging
ha-llm-train --config configs/diffusion/small.yaml \
  --no-tensorboard \
  --no-wandb
```

## Distributed Training

In distributed training (DDP/FSDP), only the main process (rank 0) logs to TensorBoard and W&B to avoid conflicts and duplicate data.

```yaml
train:
  parallel_strategy: ddp  # or 'fsdp'

logging:
  tensorboard: true
  wandb: true
  wandb_project: "distributed-training"
```

## Best Practices

1. **Use TensorBoard for local development** - Fast, no internet required
2. **Use W&B for production experiments** - Better organization, team collaboration
3. **Enable both for important runs** - Redundancy and different visualization options
4. **Use tags in W&B** - Organize experiments by model type, dataset, etc.
5. **Enable gradient tracking sparingly** - `wandb_watch_model` adds overhead
6. **Set appropriate log frequency** - Balance between detail and performance

## Troubleshooting

### TensorBoard not showing data
```bash
# Check if logs are being written
ls -la data/tb/

# Ensure correct log directory
tensorboard --logdir data/tb
```

### W&B authentication issues
```bash
# Re-login
wandb login

# Check authentication
wandb verify
```

### W&B not installed
```bash
pip install wandb
```

### Gradient tracking causing OOM
```yaml
logging:
  wandb_watch_model: false  # Disable gradient tracking
```

## API Reference

See `src/ha_llm/utils/experiment_logger.py` for the full API:

```python
from ha_llm.utils.experiment_logger import make_experiment_logger

logger = make_experiment_logger(
    tensorboard_enabled=True,
    tensorboard_dir="runs",
    wandb_enabled=True,
    wandb_project="my-project",
)

# Log scalars
logger.log_scalars(step=100, metrics={"loss": 0.5, "accuracy": 0.9})

# Log hyperparameters
logger.log_hparams({"lr": 0.001, "batch_size": 32})

# Log text
logger.log_text("config", "model: transformer", step=0)

# Log images
logger.log_image("sample", image_tensor, step=100)

# Close logger
logger.close()
```
