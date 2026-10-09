# Data format

Memento-54K uses UTF-8 JSON Lines. Each line has video_uid (an Ego4D video UUID) and conversation (a chronologically ordered array). Each turn contains role (user or assistant), content (text), time (seconds from video start), and task (task identifier). Multiple task streams can be interleaved in the same example. UID lists identify videos, not people.

| Split | Examples | Videos | User turns | Assistant turns |
|---|---:|---:|---:|---:|
| train | 53,626 | 4,426 | 151,578 | 2,517,137 |
| test | 198 | 40 | 635 | 12,829 |

There is no video overlap between the training and test splits.

The task IDs combine modality (object, action or text), reasoning category (spatial or temporal), and task type such as appear, disappear, counting, duration, ordering, changing or abnormal. Some combinations are not defined or not represented in the test split.

See the [dataset card](https://huggingface.co/datasets/liarzone/Memento-54K) for construction and video access.

Each visual cache file is named VIDEO_UID.pt, with shape [frames,10,1024] and BF16 values. The adjacent FEATURE_DIRECTORY_metadata.json maps each UID to its feature path and duration, using duration=(frames-1)/2. Rebuild this metadata after moving features to a different machine.

Video UID lists and a verification script are included under `uids/` in the dataset repository. Preprocessing uses Ego4D `video_540ss`.
