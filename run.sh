for f in voxel/*.npy; do
    uv run python visual_voxel.py --path "$f" --mode boxes
done
