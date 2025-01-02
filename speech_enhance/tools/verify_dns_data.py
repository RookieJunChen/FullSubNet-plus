import os
from pathlib import Path
import soundfile as sf
import numpy as np

def verify_dns_data():
    base_dir = Path("/home/joel/src/data/DNS-Challenge-interspeech2020-master/datasets")
    
    stats = {
        "clean": {"count": 0, "duration": 0},
        "noise": {"count": 0, "duration": 0},
        "test": {"count": 0, "duration": 0},
        "blind_test_set": {"count": 0, "duration": 0}
    }
    
    # Check each directory
    for data_type in ["clean", "noise", "test_set", "blind_test_set"]:
        dir_path = base_dir / data_type
        if not dir_path.exists():
            print(f"Warning: {data_type} directory not found at {dir_path}")
            continue
            
        files = list(dir_path.glob("*.wav"))
        key = "test" if data_type == "test_set" else data_type
        stats[key]["count"] = len(files)
        
        # Sample a few files to check duration
        for file in files[:10]:  # Check first 10 files
            try:
                info = sf.info(file)
                stats[key]["duration"] += info.duration
            except Exception as e:
                print(f"Error reading {file}: {e}")
                
    # Print statistics
    print("\nDataset Statistics:")
    print("-" * 50)
    for key, data in stats.items():
        avg_duration = data["duration"] / 10 if data["count"] > 0 else 0
        print(f"{key.title()}:")
        print(f"  - Files: {data['count']}")
        print(f"  - Avg Duration: {avg_duration:.2f}s (from 10 sample files)")
    
    return all(stats[k]["count"] > 0 for k in stats)

if __name__ == "__main__":
    if verify_dns_data():
        print("\nDataset verification passed!")
    else:
        print("\nWarning: Some data may be missing!") 