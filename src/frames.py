<<<<<<< HEAD
#!/usr/bin/env python3
"""
Extract Frames from Videos for Dataset Preparation
- Extract frames from videos
- Organize by person name (video name)
- Split into train/val/test (70/10/20)
"""

import cv2
import os
from pathlib import Path
import random
import shutil
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FrameDatasetCreator:
    def __init__(self):
        pass

    def process_video(self, video_path, person_name, output_dir, frame_interval=15):
        """Extract frames from video for a person"""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return 0

        frame_count = 0
        extracted_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process every nth frame
            if frame_count % frame_interval == 0:
                frame_filename = f"{person_name}_frame{frame_count:06d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                extracted_count += 1

            frame_count += 1

        cap.release()
        return extracted_count

    def get_person_name_from_video(self, video_path):
        """Extract person name from video filename"""
        video_name = Path(video_path).stem
        suffixes_to_remove = ['_P1', '_P2', '_P3', '_P4', '_P5', '_1', '_2', '_3', '_4', '_5']
        person_name = video_name
        for suffix in suffixes_to_remove:
            if person_name.endswith(suffix):
                person_name = person_name[:-len(suffix)]
                break
        return person_name

    def create_dataset(self, video_dir, output_dir, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
        """Create complete dataset with train/val/test split"""

        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
            raise ValueError("Train, val, and test ratios must sum to 1.0")

        train_dir = os.path.join(output_dir, 'train')
        val_dir = os.path.join(output_dir, 'val')
        test_dir = os.path.join(output_dir, 'test')

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV'}
        video_files = [f for f in Path(video_dir).glob('*') 
                      if f.suffix.lower() in video_extensions]

        if not video_files:
            logger.error(f"No video files found in {video_dir}")
            return

        logger.info(f"Found {len(video_files)} video files")

        person_videos = {}
        for video_file in video_files:
            person_name = self.get_person_name_from_video(video_file)
            if person_name not in person_videos:
                person_videos[person_name] = []
            person_videos[person_name].append(video_file)

        logger.info(f"Found {len(person_videos)} unique persons")

        total_frames = 0
        person_stats = {}

        for person_name, videos in person_videos.items():
            logger.info(f"Processing person: {person_name} ({len(videos)} videos)")

            temp_dir = os.path.join(output_dir, f'temp_{person_name}')
            os.makedirs(temp_dir, exist_ok=True)

            person_frames = []
            for video_file in videos:
                logger.info(f"  Processing video: {video_file.name}")
                extracted_count = self.process_video(str(video_file), person_name, temp_dir)
                current_frames = [f for f in Path(temp_dir).glob('*.jpg')]
                person_frames.extend(current_frames)
                logger.info(f"  Extracted {extracted_count} frames")

            if not person_frames:
                logger.warning(f"No frames extracted for {person_name}")
                shutil.rmtree(temp_dir, ignore_errors=True)
                continue

            random.shuffle(person_frames)

            n_frames = len(person_frames)
            n_train = int(n_frames * train_ratio)
            n_val = int(n_frames * val_ratio)
            n_test = n_frames - n_train - n_val

            train_frames = person_frames[:n_train]
            val_frames = person_frames[n_train:n_train + n_val]
            test_frames = person_frames[n_train + n_val:]

            for split_name, split_frames, split_dir in [
                ('train', train_frames, train_dir),
                ('val', val_frames, val_dir),
                ('test', test_frames, test_dir)
            ]:
                if split_frames:
                    person_split_dir = os.path.join(split_dir, person_name)
                    os.makedirs(person_split_dir, exist_ok=True)

                    for frame_file in split_frames:
                        try:
                            if frame_file.exists():
                                shutil.move(str(frame_file), os.path.join(person_split_dir, frame_file.name))
                        except Exception as e:
                            logger.warning(f"Could not move {frame_file}: {e}")
                            continue

            shutil.rmtree(temp_dir, ignore_errors=True)

            person_stats[person_name] = {
                'total_frames': n_frames,
                'train_frames': len(train_frames),
                'val_frames': len(val_frames),
                'test_frames': len(test_frames)
            }
            total_frames += n_frames

        logger.info("\n" + "="*50)
        logger.info("DATASET CREATION SUMMARY")
        logger.info("="*50)
        logger.info(f"Total persons: {len(person_stats)}")
        logger.info(f"Total frames: {total_frames}")
        logger.info(f"Output directory: {output_dir}")

        logger.info("\nPer-person statistics:")
        for person, stats in person_stats.items():
            logger.info(f"  {person}: {stats['total_frames']} frames "
                       f"(train: {stats['train_frames']}, "
                       f"val: {stats['val_frames']}, "
                       f"test: {stats['test_frames']})")

        logger.info("\nDirectory structure:")
        logger.info(f"  {output_dir}/")
        logger.info(f"  ├── train/")
        logger.info(f"  │   ├── person1/")
        logger.info(f"  │   ├── person2/")
        logger.info(f"  │   └── ...")
        logger.info(f"  ├── val/")
        logger.info(f"  │   ├── person1/")
        logger.info(f"  │   ├── person2/")
        logger.info(f"  │   └── ...")
        logger.info(f"  └── test/")
        logger.info(f"      ├── person1/")
        logger.info(f"      ├── person2/")
        logger.info(f"      └── ...")

        logger.info("\nFrame extraction completed!")

def main():
    """Main function - just run the script"""

    # =====================
    # 4. Cấu hình thư mục
    # =====================
    VIDEO_DIR = "../VIDEO"
    OUTPUT_DIR = "../FRAME_DATASET"

    if not os.path.exists(VIDEO_DIR):
        logger.error(f"Video directory '{VIDEO_DIR}' not found!")
        logger.info("Please create a 'VIDEO' directory and put your video files there.")
        logger.info("Video files should be named as person names (e.g., 'John.mp4', 'Mary_1.mp4')")
        return

    creator = FrameDatasetCreator()

    logger.info("Starting frame dataset creation...")
    logger.info(f"Video directory: {VIDEO_DIR}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info("Split ratios: Train=70%, Val=10%, Test=20%")

    try:
        creator.create_dataset(
            video_dir=VIDEO_DIR,
            output_dir=OUTPUT_DIR,
            train_ratio=0.7,
            val_ratio=0.1,
            test_ratio=0.2
        )

        logger.info("\n✅ Frame extraction completed successfully!")
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        return

if __name__ == "__main__":
=======
#!/usr/bin/env python3
"""
Extract Frames from Videos for Dataset Preparation
- Extract frames from videos
- Organize by person name (video name)
- Split into train/val/test (70/10/20)
"""

import cv2
import os
from pathlib import Path
import random
import shutil
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FrameDatasetCreator:
    def __init__(self):
        pass

    def process_video(self, video_path, person_name, output_dir, frame_interval=15):
        """Extract frames from video for a person"""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return 0

        frame_count = 0
        extracted_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process every nth frame
            if frame_count % frame_interval == 0:
                frame_filename = f"{person_name}_frame{frame_count:06d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                extracted_count += 1

            frame_count += 1

        cap.release()
        return extracted_count

    def get_person_name_from_video(self, video_path):
        """Extract person name from video filename"""
        video_name = Path(video_path).stem
        suffixes_to_remove = ['_P1', '_P2', '_P3', '_P4', '_P5', '_1', '_2', '_3', '_4', '_5']
        person_name = video_name
        for suffix in suffixes_to_remove:
            if person_name.endswith(suffix):
                person_name = person_name[:-len(suffix)]
                break
        return person_name

    def create_dataset(self, video_dir, output_dir, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
        """Create complete dataset with train/val/test split"""

        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
            raise ValueError("Train, val, and test ratios must sum to 1.0")

        train_dir = os.path.join(output_dir, 'train')
        val_dir = os.path.join(output_dir, 'val')
        test_dir = os.path.join(output_dir, 'test')

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV'}
        video_files = [f for f in Path(video_dir).glob('*') 
                      if f.suffix.lower() in video_extensions]

        if not video_files:
            logger.error(f"No video files found in {video_dir}")
            return

        logger.info(f"Found {len(video_files)} video files")

        person_videos = {}
        for video_file in video_files:
            person_name = self.get_person_name_from_video(video_file)
            if person_name not in person_videos:
                person_videos[person_name] = []
            person_videos[person_name].append(video_file)

        logger.info(f"Found {len(person_videos)} unique persons")

        total_frames = 0
        person_stats = {}

        for person_name, videos in person_videos.items():
            logger.info(f"Processing person: {person_name} ({len(videos)} videos)")

            temp_dir = os.path.join(output_dir, f'temp_{person_name}')
            os.makedirs(temp_dir, exist_ok=True)

            person_frames = []
            for video_file in videos:
                logger.info(f"  Processing video: {video_file.name}")
                extracted_count = self.process_video(str(video_file), person_name, temp_dir)
                current_frames = [f for f in Path(temp_dir).glob('*.jpg')]
                person_frames.extend(current_frames)
                logger.info(f"  Extracted {extracted_count} frames")

            if not person_frames:
                logger.warning(f"No frames extracted for {person_name}")
                shutil.rmtree(temp_dir, ignore_errors=True)
                continue

            random.shuffle(person_frames)

            n_frames = len(person_frames)
            n_train = int(n_frames * train_ratio)
            n_val = int(n_frames * val_ratio)
            n_test = n_frames - n_train - n_val

            train_frames = person_frames[:n_train]
            val_frames = person_frames[n_train:n_train + n_val]
            test_frames = person_frames[n_train + n_val:]

            for split_name, split_frames, split_dir in [
                ('train', train_frames, train_dir),
                ('val', val_frames, val_dir),
                ('test', test_frames, test_dir)
            ]:
                if split_frames:
                    person_split_dir = os.path.join(split_dir, person_name)
                    os.makedirs(person_split_dir, exist_ok=True)

                    for frame_file in split_frames:
                        try:
                            if frame_file.exists():
                                shutil.move(str(frame_file), os.path.join(person_split_dir, frame_file.name))
                        except Exception as e:
                            logger.warning(f"Could not move {frame_file}: {e}")
                            continue

            shutil.rmtree(temp_dir, ignore_errors=True)

            person_stats[person_name] = {
                'total_frames': n_frames,
                'train_frames': len(train_frames),
                'val_frames': len(val_frames),
                'test_frames': len(test_frames)
            }
            total_frames += n_frames

        logger.info("\n" + "="*50)
        logger.info("DATASET CREATION SUMMARY")
        logger.info("="*50)
        logger.info(f"Total persons: {len(person_stats)}")
        logger.info(f"Total frames: {total_frames}")
        logger.info(f"Output directory: {output_dir}")

        logger.info("\nPer-person statistics:")
        for person, stats in person_stats.items():
            logger.info(f"  {person}: {stats['total_frames']} frames "
                       f"(train: {stats['train_frames']}, "
                       f"val: {stats['val_frames']}, "
                       f"test: {stats['test_frames']})")

        logger.info("\nDirectory structure:")
        logger.info(f"  {output_dir}/")
        logger.info(f"  ├── train/")
        logger.info(f"  │   ├── person1/")
        logger.info(f"  │   ├── person2/")
        logger.info(f"  │   └── ...")
        logger.info(f"  ├── val/")
        logger.info(f"  │   ├── person1/")
        logger.info(f"  │   ├── person2/")
        logger.info(f"  │   └── ...")
        logger.info(f"  └── test/")
        logger.info(f"      ├── person1/")
        logger.info(f"      ├── person2/")
        logger.info(f"      └── ...")

        logger.info("\nFrame extraction completed!")

def main():
    """Main function - just run the script"""

    # =====================
    # 4. Cấu hình thư mục
    # =====================
    VIDEO_DIR = "../VIDEO"
    OUTPUT_DIR = "../FRAME_DATASET"

    if not os.path.exists(VIDEO_DIR):
        logger.error(f"Video directory '{VIDEO_DIR}' not found!")
        logger.info("Please create a 'VIDEO' directory and put your video files there.")
        logger.info("Video files should be named as person names (e.g., 'John.mp4', 'Mary_1.mp4')")
        return

    creator = FrameDatasetCreator()

    logger.info("Starting frame dataset creation...")
    logger.info(f"Video directory: {VIDEO_DIR}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info("Split ratios: Train=70%, Val=10%, Test=20%")

    try:
        creator.create_dataset(
            video_dir=VIDEO_DIR,
            output_dir=OUTPUT_DIR,
            train_ratio=0.7,
            val_ratio=0.1,
            test_ratio=0.2
        )

        logger.info("\n✅ Frame extraction completed successfully!")
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        return

if __name__ == "__main__":
>>>>>>> 0de0a2c (update)
    main()