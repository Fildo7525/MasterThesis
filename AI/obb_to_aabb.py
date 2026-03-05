from pathlib import Path

def obb_to_aabb(obb_label_dir: Path, aabb_label_dir: Path):
    """
    Convert OBB labels (class x1 y1 x2 y2 x3 y3 x4 y4) 
    to AABB labels (class cx cy w h) by taking the bounding box of the 4 corners.
    """
    aabb_label_dir.mkdir(parents=True, exist_ok=True)
    
    for obb_file in obb_label_dir.glob("*.txt"):
        aabb_file = aabb_label_dir / obb_file.name
        
        with open(obb_file) as f_in, open(aabb_file, "w") as f_out:
            for line in f_in:
                parts = line.strip().split()
                if not parts:
                    continue
                
                if len(parts) == 9:  # OBB format
                    class_id = parts[0]
                    coords = list(map(float, parts[1:]))
                    
                    xs = coords[0::2]  # x1, x2, x3, x4
                    ys = coords[1::2]  # y1, y2, y3, y4
                    
                    minx, maxx = min(xs), max(xs)
                    miny, maxy = min(ys), max(ys)
                    
                    cx = (minx + maxx) / 2
                    cy = (miny + maxy) / 2
                    w  = maxx - minx
                    h  = maxy - miny
                    
                    f_out.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                
                elif len(parts) == 5:  # already AABB
                    f_out.write(line)
                
                else:
                    print(f"Warning: skipping unexpected format in {obb_file}: {line.strip()}")

obb_dir  = Path(f"/home/samuel/test/MasterThesis/Orthomosaics/large/augmented/horizontal/labels_obb")
aabb_dir = Path(f"/home/samuel/test/MasterThesis/Orthomosaics/old_data_AABB/large/augmented/horizontal/labels_aabb")
obb_to_aabb(obb_dir, aabb_dir)
print(f"Converted OBB labels from {obb_dir} to AABB labels in {aabb_dir}")