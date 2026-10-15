# Scene Graph Evaluation on PVSG (VidOR)

## 1. Set Up the Environment

Set up the Conda environment according to the environment configuration and dependencies specified in the **VideoSeek paper/repository**.

Activate the environment before running the evaluation.

## 2. Download the VidOR Dataset

Download the **VidOR dataset** from the PVSG dataset's SharePoint/OneDrive folder:

[Download PVSG VidOR Dataset](https://entuedu-my.sharepoint.com/personal/jingkang001_e_ntu_edu_sg/_layouts/15/onedrive.aspx?id=/personal/jingkang001_e_ntu_edu_sg/Documents/PVSG_dataset/VidOR&viewid=e1cc98a2-1a48-4cb1-ac58-6f4cfa816987&ga=1)

Save the downloaded dataset under the following directory:

```text
svg2_dataset/
└── vidor/
    └── videos/
```

Ensure the video files are placed in the expected directory structure before running evaluation.

The ground-truth annotation file should be available at:

```text
svg2_dataset/PVSG dataset.json
```

## 3. Configure NVIDIA NIM API Access

Obtain an API key from [NVIDIA NIM](https://build.nvidia.com/) for model inference.

Configure the API key using the environment variable expected by the VideoSeek implementation. Do not commit API keys or other credentials to version control.

## 4. Run Scene Graph Evaluation

Run the following command from the project directory:

```bash
python evaluate_scenegraph.py \
  --video_dir "svg2_dataset/vidor/videos" \
  --num_samples 5 \
  --gt_file "svg2_dataset/PVSG dataset.json" \
  --output_dir output/scenegraph_batch_eval \
  --max_steps 10 \
  --lenient_semantic \
  --tiou_thresholds "0.1,0.3,0.5" \
  --verbose
```

### Evaluation Configuration

| Argument             | Description                                                                  |
| -------------------- | ---------------------------------------------------------------------------- |
| `--video_dir`        | Directory containing VidOR video files.                                      |
| `--num_samples 5`    | Evaluate 5 samples.                                                          |
| `--gt_file`          | Path to the PVSG ground-truth annotation file.                               |
| `--output_dir`       | Directory in which evaluation outputs are saved.                             |
| `--max_steps 10`     | Maximum number of agent steps.                                               |
| `--lenient_semantic` | Enable lenient semantic matching during evaluation.                          |
| `--tiou_thresholds`  | Evaluate temporal relation matching at tIoU thresholds of 0.1, 0.3, and 0.5. |
| `--verbose`          | Enable detailed evaluation logging.                                          |

## 5. Outputs

Evaluation results will be saved under:

```text
output/scenegraph_batch_eval/
```

Review the generated results and logs for object-label precision and recall, relation/triplet precision and recall, and temporal IoU-based matching performance.
